class Logger:
    def __init__(self, frame_limit, frame_threshold):
        self.matches = [] #array for matches, sorted chronologically. if len > frame_limit, remove index 0
        for i in range(0,frame_limit):
            self.matches.append("")
        self.frame_limit = frame_limit
        self.frame_threshold = frame_threshold

    def log(self, best_match):
        if best_match:
            self.matches.append(best_match)

        if len(self.matches) > self.frame_limit: #remove index 0 which is chronologically last frame
            self.matches.pop(0)

        match_frequencies = {} #dict {match : frequency}

        for i in range(self.frame_limit-1, -1, -1): #iteration from most recent frame -> last
            current_match = self.matches[i]
            if current_match not in match_frequencies.keys(): #if match id doesn't exist, set frequency to 1
                match_frequencies.update({current_match : 1})
            else: #otherwise increment frequency
                match_frequencies.update({current_match : match_frequencies.get(current_match)+1})
                if match_frequencies.get(current_match) >= self.frame_threshold: #if frequency == threshold, detect
                    self.matches = [""] * self.frame_limit
                    return current_match
        return ""
