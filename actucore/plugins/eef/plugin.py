"""Manual relative TCP and gripper tests over the two TRON2 EEF ports."""
import json
import math
import threading
import time
from uuid import uuid4

from .. import control_stream


_SCHEMA = "motus.eef/1"
_FRAME = "base_Link"
_CONTROL = "/limx_tron2/control/eef"
_TRANSLATION = ("dx_m", "dy_m", "dz_m")
_ROTATION = ("rx_rad", "ry_rad", "rz_rad")
_DELTAS = (*_TRANSLATION, *_ROTATION, "opening_delta_m")


def _now_ms():
    return time.time_ns() // 1_000_000


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _control_limits(descriptor):
    """Verify the measured 16-value layout against the wired command consumer."""
    layout = [("eef_l", 0, 7, "m+quat_xyzw"), ("gripper_l", 7, 1, "m"),
              ("eef_r", 8, 7, "m+quat_xyzw"), ("gripper_r", 15, 1, "m")]
    if (not isinstance(descriptor, dict) or descriptor.get("control_interface") != control_stream.SCHEMA
            or descriptor.get("mode") != "eef_pose" or descriptor.get("dof") != 16
            or descriptor.get("frame") != _FRAME):
        raise ValueError("Connect a motus.control/1 dual TCP and gripper actuator in base_Link")
    state = descriptor.get("state") or {}
    if not isinstance(state, dict) or state.get("schema") != _SCHEMA or state.get("field") != "eef":
        raise ValueError("The actuator must declare motus.eef/1 feedback in eef")
    groups = descriptor.get("groups") or []
    if (not isinstance(groups, list) or not all(isinstance(g, dict) for g in groups)
            or [(g.get("name"), g.get("offset"), g.get("count"), g.get("unit")) for g in groups] != layout
            or any(g.get("advisory") for g in groups)):
        raise ValueError("The actuator must execute left pose/opening, then right pose/opening")
    limits = descriptor.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("Missing actuator limits")
    bounds = [limits.get(k) for k in ("lower", "upper")]
    if (not all(isinstance(b, (list, tuple)) and len(b) == 16 and all(map(_number, b)) for b in bounds)
            or any(lo > hi for lo, hi in zip(*bounds))):
        raise ValueError("Expected 16 finite ordered actuator bounds")
    rate = descriptor.get("rate")
    if (not isinstance(rate, dict)
            or not all(_number(rate.get(k)) and rate[k] > 0 for k in
                       ("expected_hz", "watchdog_ms", "max_obs_age_ms"))):
        raise ValueError("Missing control rate, watchdog or observation age")
    return bounds


def _translate(start, delta, lower, upper):
    """Clip the segment at the first Cartesian box face, preserving its direction."""
    goal = [a + d for a, d in zip(start, delta)]
    if all(lo <= v <= hi for v, lo, hi in zip(goal, lower, upper)):
        return goal
    if not all(lo <= v <= hi for v, lo, hi in zip(start, lower, upper)):
        raise ValueError("TCP is outside the workspace; the requested target must be inside")
    fraction = 1.
    for a, d, v, lo, hi in zip(start, delta, goal, lower, upper):
        if v < lo or v > hi:
            fraction = min(fraction, ((lo if v < lo else hi) - a) / d)
    return [min(hi, max(lo, a + d * fraction)) for a, d, lo, hi in zip(start, delta, lower, upper)]


def _rotate(quaternion, rotvec):
    """Left multiplication applies the rotation about the TCP in base-frame axes."""
    angle = math.hypot(*rotvec)
    scale = math.sin(angle / 2) / angle
    x, y, z = (v * scale for v in rotvec)
    w = math.cos(angle / 2)
    a, b, c, d = quaternion
    q = (w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
         w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c)
    norm = math.hypot(*q)
    return [v / norm for v in q]


class EefPlugin:
    PREFIX = "eef"

    def __init__(self, plugin_cfg, executor):
        self._executor = executor
        self._control_topic = plugin_cfg.get("control_topic") or _CONTROL
        self._lock = threading.RLock()
        self._source = f"mcp__actucore__eef/{uuid4().hex}"
        self._seq = 0
        self._descriptor = self._bounds = None
        self._node = self._state_topic = self._pub = None
        self._sample = self._target = self._last = None

    def _ports(self):
        source = {"format": "state/eef"}
        if self._state_topic:
            source["topic"] = self._state_topic
        return {"topic_in": [source], "topic_out": [{"format": "control/eef", "topic": self._control_topic}]}

    def get_tools(self):
        descriptions = ("前为正，后为负（米）", "左为正，右为负（米）", "上为正，下为负（米）",
                        "绕基座 X 轴的旋转向量分量（弧度）", "绕基座 Y 轴的旋转向量分量（弧度）",
                        "绕基座 Z 轴的旋转向量分量（弧度）", "张开为正，闭合为负（米）")
        fields = {name: {"type": "number", "description": desc} for name, desc in zip(_DELTAS, descriptions)}
        return [{"name": "eef", "type": "processor", "multiInstance": False, **self._ports(),
                 "description": "TRON2 末端测试。空闲时持续发送当前实测值，移动时持续发送绝对目标。",
                 "inputSchema": {"type": "object", "required": ["action"], "properties": {
                     "action": {"type": "string", "enum": ["info", "start", "stop", "move_by", "cancel"]},
                     "arm": {"type": "string", "enum": ["left", "right"]},
                     "control_interface": {"type": "object"}, **fields},
                     "x-action-params": {"move_by": {"params": ["arm", *_DELTAS],
                                                     "description": "从当前实测帧执行一次相对移动"},
                                         "cancel": {"params": [], "description": "将最新实测的双臂位姿和开口设为保持目标"}},
                     "x-hooks": {"on_interrupt_motion": {"action": "cancel"}, "on_interrupt_all": {"action": "cancel"}},
                     "x-is-dangerous": True, "x-resource": ["arm_l", "arm_r", "gripper_l", "gripper_r"]}}]

    def dispatch(self, name, args):
        if name != self.PREFIX:
            return None
        with self._lock:
            try:
                action = args.get("action", name)
                if action == "start":
                    self._start(args)
                elif action == "stop":
                    self._stop()
                elif action == "move_by":
                    return self._move(args)
                elif action == "cancel":
                    if self._node:
                        if self._target is not None:
                            arm, group, _ = self._target
                            self._target = (arm, group, None)
                        self._last = None
                        if self._sample and self._fresh(self._sample["stamp_ms"]):
                            return self._publish()
                        return {"state": "running", "status": "waiting_for_state"}
                elif action != "info":
                    raise ValueError("Unknown eef action")
                return {"state": "running" if self._node else "idle", **self._ports(),
                        "sample": self._sample, "last": self._last}
            except (ValueError, TypeError, RuntimeError, OverflowError) as error:
                return {"state": "error", "message": str(error)}

    def _start(self, args):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        topics = args.get("input_topics", [])
        if not isinstance(topics, list):
            raise ValueError("Expected a list of state/eef input topics")
        topics = topics + ([args["input_topic"]] if args.get("input_topic") else [])
        if (not topics or any(not isinstance(t, str) or not t.startswith("/") for t in topics)
                or len(set(topics)) != 1 or topics[0] == self._control_topic):
            raise ValueError("Connect one state/eef input, distinct from the control/eef output")
        descriptor = args.get("control_interface")
        if self._node:
            if self._state_topic != topics[0] or (descriptor is not None and descriptor != self._descriptor):
                raise ValueError("Stop eef before changing its connections")
            return
        bounds = _control_limits(descriptor)
        node = Node("eef")
        try:
            qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            pub = node.create_publisher(String, self._control_topic, qos)
            node.create_subscription(String, topics[0], lambda msg: self._receive(node, msg), qos)
            self._executor.add_node(node)
        except Exception:
            node.destroy_node()
            raise
        self._node, self._state_topic, self._pub = node, topics[0], pub
        self._descriptor, self._bounds = descriptor, bounds
        self._ttl = control_stream.ttl_ms(descriptor["rate"]["expected_hz"], descriptor)

    def _stop(self):
        if self._node:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        self._node = self._state_topic = self._pub = None
        self._sample = self._target = self._last = None
        self._descriptor = self._bounds = None

    def _fresh(self, stamp):
        return type(stamp) is int and -100 <= _now_ms() - stamp <= self._descriptor["rate"]["max_obs_age_ms"]

    def _receive(self, node, msg):
        try:
            data = json.loads(msg.data)
            if (not isinstance(data, dict) or data.keys() != {"schema", "frame", "stamp_ms", "eef"}
                    or data["schema"] != _SCHEMA or data["frame"] != _FRAME):
                return
            stamp, eef = data["stamp_ms"], data["eef"]
            if (not isinstance(eef, list) or len(eef) != 16
                    or not all(map(_number, eef)) or any(abs(math.hypot(*eef[i:i + 4]) - 1.) > .001
                                                         for i in (3, 11)) or min(eef[7], eef[15]) < 0):
                return
        except (ValueError, TypeError, OverflowError):
            return
        with self._lock:
            if node is not self._node or not self._fresh(stamp):
                return
            if self._sample and stamp < self._sample["stamp_ms"]:
                return
            self._sample = data
            self._publish()

    def _publish(self):
        from std_msgs.msg import String

        observed = self._sample["stamp_ms"]
        values = self._sample["eef"].copy()
        arm, origin = None, observed
        if self._target is not None:
            arm, group, goal = self._target
            if goal is None:  # Capture the cancellation point once, even after feedback loss.
                goal = values[group]
                self._target = (arm, group, goal)
            values[group] = goal
            if self._last:
                origin = self._last["state_stamp_ms"]
        count = self._last["publish_count"] if self._last and self._last["state_stamp_ms"] == origin else 0
        self._seq += 1
        command = control_stream.build(seq=self._seq, values=values, mode="eef_pose", dof=len(values),
                                       source=self._source, stamp_ms=_now_ms(), obs_stamp_ms=observed,
                                       ttl_ms=self._ttl)
        command["frame"] = _FRAME
        self._last = {"arm": arm, "state_stamp_ms": origin, "command": command, "publish_count": count}
        msg = String()
        msg.data = json.dumps(command, allow_nan=False)
        try:
            self._pub.publish(msg)
        except RuntimeError as error:
            self._last.update(status="publish_error", message=str(error))
        else:
            self._last.update(status="published", publish_count=count + 1)
        return {"state": "running", **self._last}

    def _move(self, args):
        if not self._sample or not self._fresh(self._sample["stamp_ms"]):
            raise ValueError("Start eef and wait for fresh eef_state feedback")
        arm = args.get("arm")
        if arm not in ("left", "right"):
            raise ValueError("Select left or right arm")
        delta = [args.get(k, 0.) for k in _DELTAS]
        if not all(map(_number, delta)) or not any(k in args for k in _DELTAS):
            raise ValueError("Provide at least one finite increment")
        translation, rotation, opening = delta[:3], delta[3:6], delta[6]
        i = 0 if arm == "left" else 8
        target = self._sample["eef"].copy()
        lower, upper = self._bounds
        if any(translation):
            target[i:i + 3] = _translate(target[i:i + 3], translation, lower[i:i + 3], upper[i:i + 3])
        if any(rotation):
            target[i + 3:i + 7] = _rotate(target[i + 3:i + 7], rotation)
        if opening:
            target[i + 7] = min(upper[i + 7], max(lower[i + 7], target[i + 7] + opening))
        if not all(map(_number, target)):
            raise ValueError("Absolute target must be finite")
        pose_changed = any(translation) or any(rotation)
        if pose_changed or opening:
            group = slice(i if pose_changed else i + 7, i + (8 if opening else 7))
            self._target = (arm, group, target[group])
        else:
            self._target = None
        self._last = None
        return self._publish()
