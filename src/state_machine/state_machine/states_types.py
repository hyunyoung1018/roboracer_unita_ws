import enum


class StateType(enum.Enum):
    """The value is the string published in BehaviorStrategy.state.

    RACELINE was called GB_TRACK. "GB" was short for gb_optimizer,
    the package that produced the global trajectory; ours is `raceline`, and
    the state means "following the raceline", so the old name pointed at
    something this workspace does not have. The controller only ever compares
    against START / TRAILING / OVERTAKE / FTGONLY, so nothing outside this
    package reads the changed string.
    """

    RACELINE = 'RACELINE'
    TRAILING = 'TRAILING'
    OVERTAKE = 'OVERTAKE'
    FTGONLY = 'FTGONLY'
    RECOVERY = 'RECOVERY'
    ATTACK = 'ATTACK'
    START = 'START'
    LOSTLINE = 'LOSTLINE'
