from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans = [[] for _ in numbers]
        max_ln = max(map(lambda x: len(x), vocab.keys()))
        for i, number in enumerate(numbers):
            str_number = str(number)

            l = 0
            r = min(l + max_ln, len(str_number))

            while l < r <= len(str_number):
                if str_number[l:r] in vocab:
                    ans[i].append(str_number[l:r])
                    l = r
                    r = min(l + max_ln, len(str_number))
                else:
                    r -= 1
        return ans


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        # return len(self.tokenize_numbers(text, vocab))
        ans = []
        max_ln = max(map(lambda x: len(x), vocab.keys()))
        
        l = 0
        r = min(l + max_ln, len(text))
        while l < r <= len(text):
            if text[l:r] in vocab:
                ans.append(text[l:r])
                l = r
                r = min(l + max_ln, len(text))
            else:
                r -= 1
        return len(ans)
        
    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        
        return round(self.count_tokens(text, vocab) / len(text.split()), 4)
