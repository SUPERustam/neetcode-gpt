from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        ans = []
        for _ in range(num_merges):
            # find
            pair_dct = {}
            for i in range(1, len(tokens)):
                pair = (tokens[i - 1], tokens[i])
                pair_dct[pair] = pair_dct.get(pair, 0) + 1
            
            if not pair_dct:
                break
            
            max_freq = max(pair_dct.values())
            candidates = sorted(p for p, c in pair_dct.items() if c == max_freq)
            best_pair = candidates[0]

            ans.append(best_pair)
            
            # merge
            i = 1
            while i < len(tokens):
                if best_pair == (tokens[i - 1], tokens[i]):
                    del tokens[i - 1]
                    tokens[i - 1] = best_pair[0] + best_pair[1]
                i += 1
                
        
        return ans

