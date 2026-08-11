from f110_msgs.msg import Obstacle, ObstacleArray


def test_message_fields_are_not_reconstructed():
    """The routing contract is object preservation, not a field allow-list."""
    msg = ObstacleArray()
    static = Obstacle(id=7, is_static=True, vs_var=1.25, is_visible=False)
    dynamic = Obstacle(id=9, is_static=False, vd_var=2.5, is_visible=True)
    msg.obstacles = [static, dynamic]

    static_result = [obs for obs in msg.obstacles if obs.is_static]
    dynamic_result = [obs for obs in msg.obstacles if not obs.is_static]

    assert static_result[0] is static
    assert dynamic_result[0] is dynamic
    assert static_result[0].vs_var == 1.25
    assert dynamic_result[0].vd_var == 2.5
