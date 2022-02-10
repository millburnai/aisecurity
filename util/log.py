class Logger:
    def __init__(self, frame_limit: int, frame_threshold) -> None:
        # NOTE: Array for matches, sorted chronologically.
        # if len > frame_limit, remove index 0
        self.matches = []
        for i in range(frame_limit):
            self.matches.append("")

        self.frame_limit = frame_limit
        self.frame_threshold = frame_threshold

    def log(self, best_match) -> str:
        best_match = str(best_match)
        self.matches.append(best_match)

        # NOTE: Remove index 0 which is chronologically last frame
        if len(self.matches) > self.frame_limit:
            self.matches.pop(0)

        # dict{match : frequency}
        match_frequencies = {}

        # NOTE: Iteration from most recent frame -> last
        for i in range(self.frame_limit - 1, -1, -1):
            current_match = self.matches[i]
            # NOTE: If match id doesn't exist, set frequency to 1
            if current_match not in match_frequencies.keys():
                match_frequencies.update({current_match: 1})

            # NOTE: Increment frequency
            else:
                match_frequencies.update(
                    {current_match: match_frequencies.get(current_match) + 1}
                )
                # NOTE: if frequency == threshold, detect
                if match_frequencies.get(current_match) >= self.frame_threshold:
                    return current_match

        return ""
