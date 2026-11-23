"""Problems a person can fix by changing images or settings."""


class UserFacingError(ValueError):
    """An error whose message is safe to show directly in the app."""


class DatasetError(UserFacingError):
    """The images or labels cannot be split and trained as they stand."""


class TrainingError(UserFacingError):
    """A run stopped because a setting is unsafe or the loss diverged."""
