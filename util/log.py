class Logger:

    def __init__(self, frame_limit, frame_threshold):
        self.matches = [None for _ in range(frame_limit)]
        # array for matches, sorted chronologically.
        # if len > frame_limit, remove index 0
        self.frame_limit = frame_limit
        self.frame_threshold = frame_threshold

    def log(self, best_match):
        if best_match:
            self.matches.append(best_match)
            if len(self.matches) > self.frame_limit:
                # remove index 0 which is chronologically last frame
                self.matches.pop(0)

        # dict {match : frequency}
        match_frequencies = {}

        # iteration from most recent frame -> last
        for i, match in enumerate(reversed(self.matches)):
            if not match:
                continue
            try:
                match_frequencies[match] += 1
                if match_frequencies[match] >= self.frame_threshold:
                    # if frequency == threshold, detect
                    self.matches = [None for _ in range(self.frame_limit)]
                    print(f"[DEBUG] logged '{match}'")
                    return match
            except KeyError:
                # if match id doesn't exist, set freq to 1; otherwise, increment
                match_frequencies[match] = 1
