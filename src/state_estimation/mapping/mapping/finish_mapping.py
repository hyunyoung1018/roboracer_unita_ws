"""
Finish a Cartographer mapping run and save the generated map.

When mapping is complete, enter `q` in the terminal running this node and press
Enter. The following files are written:

    maps/<map_name>/<map_name>.pbstream
    maps/<map_name>/<map_name>.png
    maps/<map_name>/<map_name>.yaml

The map can also be saved manually with:

    ros2 service call /finish_mapping std_srvs/srv/Trigger {}
"""

import os
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from cartographer_ros_msgs.srv import FinishTrajectory, WriteState
from nav2_msgs.srv import SaveMap
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger

from .paths import is_inside_install_tree, resolve_maps_source_dir


# cartographer_ros_msgs/StatusResponse에서 0은 성공을 의미합니다.
STATUS_OK = 0

# trajectory 종료 후 pose graph 최적화를 기다리는 시간입니다.
SETTLE_SEC = 3.0

# ROS 서비스가 나타날 때까지 기다리는 최대 시간입니다.
SERVICE_TIMEOUT_SEC = 10.0


class FinishMapping(Node):

    def __init__(self):
        super().__init__('finish_mapping')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        self.declare_parameter('map_name', '')
        self.declare_parameter('maps_dir', '')
        self.declare_parameter('map_dir', '')
        self.declare_parameter('trajectory_id', 0)
        self.declare_parameter('save_pbstream', True)
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('image_format', 'png')
        self.declare_parameter('map_mode', 'trinary')
        self.declare_parameter('free_thresh', 0.196)
        self.declare_parameter('occupied_thresh', 0.65)

        self.map_name = self.get_parameter('map_name').value

        if not self.map_name:
            raise RuntimeError('map_name parameter is required')

        # 저장 중 다른 서비스 응답을 동시에 처리할 수 있도록 사용합니다.
        self.cb_group = ReentrantCallbackGroup()

        # ------------------------------------------------------------------
        # Service clients
        # ------------------------------------------------------------------

        self.cli_finish = self.create_client(
            FinishTrajectory,
            '/finish_trajectory',
            callback_group=self.cb_group,
        )

        self.cli_write_state = self.create_client(
            WriteState,
            '/write_state',
            callback_group=self.cb_group,
        )

        self.cli_save_map = self.create_client(
            SaveMap,
            '/map_saver/save_map',
            callback_group=self.cb_group,
        )

        # q가 입력됐을 때 이 노드가 제공하는 /finish_mapping 서비스를
        # 내부적으로 호출하기 위한 클라이언트입니다.
        self.cli_finish_mapping = self.create_client(
            Trigger,
            '/finish_mapping',
            callback_group=self.cb_group,
        )

        # ------------------------------------------------------------------
        # Map subscription
        # ------------------------------------------------------------------

        self._last_map = None

        self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self._map_cb,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
            callback_group=self.cb_group,
        )

        # ------------------------------------------------------------------
        # Finish-mapping service
        # ------------------------------------------------------------------

        self.create_service(
            Trigger,
            '/finish_mapping',
            self._finish_cb,
            callback_group=self.cb_group,
        )

        # ------------------------------------------------------------------
        # Terminal input thread
        # ------------------------------------------------------------------

        # input()이 ROS executor를 막지 않도록 별도 스레드에서 실행합니다.
        self._input_thread = threading.Thread(
            target=self._wait_for_q,
            daemon=True,
        )
        self._input_thread.start()

        self.get_logger().info(
            f'Mapping "{self.map_name}".\n'
            f'매핑이 끝나면 이 터미널에서 q를 입력하고 Enter를 누르세요.\n'
            f'또는 다른 터미널에서 다음 명령을 실행할 수 있습니다:\n'
            f'    ros2 service call /finish_mapping '
            f'std_srvs/srv/Trigger {{}}'
        )

    # ----------------------------------------------------------------------
    # Terminal input
    # ----------------------------------------------------------------------

    def _wait_for_q(self) -> None:
        """실행 중인 터미널에서 q + Enter 입력을 기다립니다."""

        try:
            # ros2 launch가 자식 프로세스의 stdin을 전달하지 않는 경우가
            # 있으므로 현재 제어 터미널을 직접 엽니다.
            with open('/dev/tty', 'r', encoding='utf-8') as terminal:
                while rclpy.ok():
                    command = terminal.readline()

                    # 터미널이 닫힌 경우입니다.
                    if command == '':
                        return

                    command = command.strip().lower()

                    if command != 'q':
                        self.get_logger().info(
                            '맵을 저장하려면 q를 입력하고 Enter를 누르세요.'
                        )
                        continue

                    self.get_logger().info(
                        'q 입력을 감지했습니다. 맵 저장을 요청합니다.'
                    )

                    if not self.cli_finish_mapping.wait_for_service(
                        timeout_sec=SERVICE_TIMEOUT_SEC
                    ):
                        self.get_logger().error(
                            '/finish_mapping 서비스를 사용할 수 없습니다.'
                        )
                        return

                    future = self.cli_finish_mapping.call_async(
                        Trigger.Request()
                    )
                    future.add_done_callback(self._save_done)
                    return

        except OSError as exc:
            self.get_logger().error(
                f'실행 중인 터미널에서 입력을 받을 수 없습니다: {exc}'
            )

    def _save_done(self, future) -> None:
        """q 입력으로 요청한 저장 작업의 결과를 출력합니다."""

        try:
            response = future.result()

            if response is None:
                self.get_logger().error(
                    '맵 저장 서비스에서 응답을 받지 못했습니다.'
                )
                return

            if response.success:
                self.get_logger().info(
                    f'맵 저장 완료: {response.message}'
                )
            else:
                self.get_logger().error(
                    f'맵 저장 실패: {response.message}'
                )

        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'맵 저장 요청 중 오류가 발생했습니다: {exc}'
            )

    # ----------------------------------------------------------------------
    # Map callback
    # ----------------------------------------------------------------------

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._last_map = msg

    # ----------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------

    def _resolve_map_dir(self) -> str:
        explicit = self.get_parameter('map_dir').value

        if explicit:
            return explicit

        maps_dir = self.get_parameter('maps_dir').value

        if not maps_dir:
            raise RuntimeError(
                'either map_dir or maps_dir must be set'
            )

        source_maps = resolve_maps_source_dir(maps_dir)

        if is_inside_install_tree(source_maps):
            self.get_logger().warn(
                f'Could not resolve {maps_dir} back to src; writing into '
                f'the install tree at {source_maps}. The map may be lost '
                f'on the next clean build.'
            )

        return os.path.join(source_maps, self.map_name)

    # ----------------------------------------------------------------------
    # Service helper
    # ----------------------------------------------------------------------

    def _call(self, client, request, what: str):
        """ROS 서비스를 동기식으로 호출합니다."""

        if not client.wait_for_service(
            timeout_sec=SERVICE_TIMEOUT_SEC
        ):
            raise RuntimeError(
                f'{what}: service {client.srv_name} is not available. '
                f'Is the mapping launch still running?'
            )

        response = client.call(request)

        if response is None:
            raise RuntimeError(
                f'{what}: call to {client.srv_name} returned nothing'
            )

        return response

    # ----------------------------------------------------------------------
    # Finish-mapping service
    # ----------------------------------------------------------------------

    def _finish_cb(self, _request, response):
        try:
            message = self._run()

            response.success = True
            response.message = message

            self.get_logger().info(message)

        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)

            self.get_logger().error(
                f'Finishing the map failed: {exc}'
            )

        return response

    def _run(self) -> str:
        """Cartographer trajectory를 종료하고 맵 파일을 저장합니다."""

        map_dir = self._resolve_map_dir()

        # 실제 /map 메시지를 한 번도 받지 못한 상태에서는 trajectory를
        # 종료하지 않습니다. 맵이 없는 상태에서 trajectory를 종료하면
        # 다시 매핑을 계속할 수 없기 때문입니다.
        if self._last_map is None:
            raise RuntimeError(
                f'No message has arrived on '
                f'{self.get_parameter("map_topic").value}, '
                f'so there is no map to save and the trajectory has NOT '
                f'been finished. Keep driving and check /scan and TF.'
            )

        os.makedirs(map_dir, exist_ok=True)

        trajectory_id = int(
            self.get_parameter('trajectory_id').value
        )

        self.get_logger().info(
            f'Finishing trajectory {trajectory_id}...'
        )

        result = self._call(
            self.cli_finish,
            FinishTrajectory.Request(
                trajectory_id=trajectory_id
            ),
            'finish_trajectory',
        )

        if result.status.code != STATUS_OK:
            # 이미 종료된 trajectory일 수 있으므로 경고만 출력합니다.
            self.get_logger().warn(
                f'/finish_trajectory returned code '
                f'{result.status.code}: {result.status.message}'
            )

        self.get_logger().info(
            f'Waiting {SETTLE_SEC:.0f}s for the final '
            f'pose graph optimisation...'
        )

        self.get_clock().sleep_for(
            rclpy.duration.Duration(seconds=SETTLE_SEC)
        )

        written = []

        # ------------------------------------------------------------------
        # Save pbstream
        # ------------------------------------------------------------------

        if self.get_parameter('save_pbstream').value:
            pbstream = os.path.join(
                map_dir,
                f'{self.map_name}.pbstream',
            )

            result = self._call(
                self.cli_write_state,
                WriteState.Request(
                    filename=pbstream,
                    include_unfinished_submaps=True,
                ),
                'write_state',
            )

            if result.status.code != STATUS_OK:
                raise RuntimeError(
                    f'/write_state failed '
                    f'({result.status.code}): '
                    f'{result.status.message}'
                )

            written.append(os.path.basename(pbstream))

        # ------------------------------------------------------------------
        # Save PNG and YAML
        # ------------------------------------------------------------------

        map_url = os.path.join(
            map_dir,
            self.map_name,
        )

        result = self._call(
            self.cli_save_map,
            SaveMap.Request(
                map_topic=self.get_parameter(
                    'map_topic'
                ).value,
                map_url=map_url,
                image_format=self.get_parameter(
                    'image_format'
                ).value,
                map_mode=self.get_parameter(
                    'map_mode'
                ).value,
                free_thresh=float(
                    self.get_parameter(
                        'free_thresh'
                    ).value
                ),
                occupied_thresh=float(
                    self.get_parameter(
                        'occupied_thresh'
                    ).value
                ),
            ),
            'save_map',
        )

        if not result.result:
            raise RuntimeError(
                '/map_saver/save_map failed. Is anything publishing '
                f'{self.get_parameter("map_topic").value}?'
            )

        written.extend([
            f'{self.map_name}.png',
            f'{self.map_name}.yaml',
        ])

        return (
            f'Wrote {", ".join(written)} to {map_dir}. '
            f'Next: ros2 launch stack_master '
            f'raceline_generator.launch.xml '
            f'map:={self.map_name}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = FinishMapping()

    # 저장 콜백이 다른 서비스의 응답을 기다리는 동안에도 ROS 응답을
    # 처리할 수 있도록 MultiThreadedExecutor를 사용합니다.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        # Ctrl+C는 저장 기능과 연결하지 않고 정상 종료만 수행합니다.
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()