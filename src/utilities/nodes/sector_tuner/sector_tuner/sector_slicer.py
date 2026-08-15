import yaml, os, subprocess, time
import rclpy
from rclpy.node import Node
from f110_msgs.msg import WpntArray
import numpy as np
from visualization_msgs.msg import MarkerArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Arrow

from .paths import resolve_source_dir

class SectorSlicer(Node):
    """
    Node for listening to gb waypoints and running a GUI to tune the sectors s.t. a yaml can be exported for dynamic reconfigure
    """
    def __init__(self, future):
        super().__init__('sector_slicer',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=False)
        # Directory the yaml is written to - the map folder. Filename is always
        # speed_scaling.yaml.
        self.declare_parameter('save_dir', '')
        # Rebuild stack_master after writing. Needed the first time a map gets
        # sectors, because a brand new yaml has no symlink in the install tree
        # yet and would stay invisible to ros2 until the next build.
        self.declare_parameter('rebuild_on_save', True)

        self.future = future
        self.global_wpnts = None
        self.track_bounds = None

        # Written into a new map's speed_scaling.yaml. The ceiling is 1.0 (the
        # raceline's own optimised profile) and the live setpoint starts at half
        # of it, so a freshly sliced map is slow until someone raises it rather
        # than unable to go fast without an edit and a rebuild.
        self.speed_limit = 1.0
        self.speed_scaling = 0.5
        # Written alongside the scaling because the controller's L1 lookahead floor
        # is paired with it - see controller.yaml. This has to be emitted here: the
        # dump below rewrites speed_scaling.yaml whole, so a field the slicer does
        # not know about disappears every time the sectors are re-cut. 1.1 is the
        # value that goes with a 0.5 scaling.
        self.t_clip_min = 1.1
        self.glob_slider_s = 0
        self.sector_pnts = [0] #Sector always has tostart at 0
        
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.wpnt_sub = self.create_subscription(
            WpntArray,
            '/global_waypoints',
            self.global_wpnts_cb,
            10)
        self.bounds_sub = self.create_subscription(
            MarkerArray,
            '/trackbounds/markers',
            self.bounds_cb,
            10)

        self.yaml_dir = resolve_source_dir(
            self.get_parameter('save_dir').get_parameter_value().string_value)
        self.rebuild_on_save = self.get_parameter('rebuild_on_save').value
        if not self.yaml_dir:
            self.get_logger().error('save_dir parameter is required')
            raise RuntimeError('save_dir parameter is required')
        self.get_logger().info(f'Sectors will be written to {self.yaml_dir}')
        self.get_logger().info('Waiting for global waypoints...')

    def global_wpnts_cb(self, data):
        self.global_wpnts = data

    def bounds_cb(self, data):
        self.track_bounds = data

    def timer_callback(self):
        if(self.global_wpnts is None):
            return
        if(self.track_bounds is None):
            return
        
        #Select Sectors via the GUI
        self.sector_gui()
        self.get_logger().info('Selected Sector IDXs: '+str(self.sector_pnts))

        #Write sectors to yaml
        self.sectors_to_yaml()
        #Indicate that the task is done and execution of the node can be stopped
        self.future.set_result(None)

    def sector_gui(self):
        #get wpnt message in list format for plotting
        s = []
        v = []
        x = []
        y = []
        for wpnt in self.global_wpnts.wpnts:
            s.append(wpnt.s_m)
            v.append(wpnt.vx_mps)
            x.append(wpnt.x_m)
            y.append(wpnt.y_m)
        s = np.array(s)
        x = np.array(x)
        y = np.array(y)

        # plot raceline without sector points
        fig, (ax1, axslider, axselect, axfinish) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 1, 1, 1]})
        ax1.plot(x, y, "r-", linewidth=0.7)
        ax1.plot([mrk.pose.position.x for mrk in self.track_bounds.markers], [mrk.pose.position.y for mrk in self.track_bounds.markers], 'g-', linewidth=0.4)
        ax1.grid()
        ax1.set_aspect("equal", "datalim")
        ax1.set_xlabel("east in m")
        ax1.set_ylabel("north in m")
        #Plot arrow at start
        arr_par = {'x': x[0], 'dx': 10 * (x[1] - x[0]),
                   'y': y[0], 'dy': 10 * (y[1] - y[0]),
                   'color': 'gray',
                   'width': 0.5}
        ax1.add_artist(Arrow(**arr_par))
        
        #Slider stuff
        def update_s(val):
            idx = int(slider.val) - 1
            self.glob_slider_s = idx #update the global slider s
            update_map(x=x, y=y, cur_s=idx)
            fig.canvas.draw_idle()

        #Btn stuff
        def select_s(event):
            #When pressing button append the new position
            self.sector_pnts.append(self.glob_slider_s)
            update_map(x=x, y=y, cur_s=self.glob_slider_s)
        
        def finish(event):
            plt.close()

            #Sectors always end at end
            self.sector_pnts.append(len(s))

            #Eliminate duplicates if necessary
            self.sector_pnts = sorted(list(set(self.sector_pnts)))
            return 

        def update_map(x, y, cur_s):
            ax1.cla()
            ax1.plot(x, y, "r-", linewidth=0.7)
            ax1.plot([mrk.pose.position.x for mrk in self.track_bounds.markers], [mrk.pose.position.y for mrk in self.track_bounds.markers], 'g-', linewidth=0.4)
            ax1.grid()
            ax1.set_aspect("equal", "datalim")
            ax1.set_xlabel("east in m")
            ax1.set_ylabel("north in m")
            ax1.scatter(x[cur_s], y[cur_s])
            if len(self.sector_pnts) > 0:
                ax1.scatter(x[self.sector_pnts], y[self.sector_pnts], c='red')

        #Matplotlib widgets for GUI
        slider = Slider(axslider, 'S [m]', 0, len(s), valinit=0, valfmt='%d')
        slider.on_changed(update_s)

        btn_select = Button(axselect, 'Select S')
        btn_select.on_clicked(select_s)

        btn_finish = Button(axfinish, 'Done')
        btn_finish.on_clicked(finish)
        
        plt.show()

    def sectors_to_yaml(self):
        #Create yaml with default speed scaling values
        n_sectors = len(self.sector_pnts) - 1
        dict_file = {'global_limit': self.speed_limit, 'n_sectors': n_sectors}
        for i in range(0, n_sectors):
            #Add sectors with scaling field
            dict_file['Sector' + str(i)] = {'start':self.sector_pnts[i] if i == 0 else self.sector_pnts[i] + 1,
                                            'end':self.sector_pnts[i+1],
                                            'scaling':self.speed_scaling,
                                            't_clip_min':self.t_clip_min}
            #Add only_FTG field to sector
            dict_file['Sector' + str(i)].update({'only_FTG': False})
            #Add no_FTG field to sector
            dict_file['Sector' + str(i)].update({'no_FTG': False})
        ros_yaml_preamble = {'sector_tuner': {'ros__parameters': dict_file}}
        
        #Save yaml to the respective maps folder
        os.makedirs(self.yaml_dir, exist_ok=True)
        yaml_path = os.path.join(self.yaml_dir, 'speed_scaling.yaml')
        with open(yaml_path, 'w') as file:
            self.get_logger().info('Dumping to {}: {}'.format(yaml_path, ros_yaml_preamble))
            yaml.dump(ros_yaml_preamble, file, sort_keys=False)

        if self.rebuild_on_save:
            # NOTE: ot_sector_slicer does the same thing. If both finish at the
            # same moment the two colcon builds collide. In practice each waits
            # on its own GUI so they finish apart; set rebuild_on_save:=false on
            # one of them and rebuild by hand if you do hit it.
            self.get_logger().info('Invoking colcon build on stack_master so the new yaml is installed.')
            subprocess.Popen("ros2 run sector_tuner finish_sector.sh", shell=True)

        self.get_logger().info("Done Slicing")

def main():
    rclpy.init()
    future = rclpy.Future()
    node = SectorSlicer(future)
    rclpy.spin_until_future_complete(node, future)
    rclpy.shutdown()
