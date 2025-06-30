
from transformers import StoppingCriteria, StoppingCriteriaList

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_ids):
        self.tokenizer = tokenizer
        self.stop_ids = stop_ids  # 리스트로 여러 개 가능

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids: (batch, sequence)
        # 마지막 토큰이 stop_ids 중 하나면 True 반환 → 멈춤
        for stop_id in self.stop_ids:
            if input_ids[0, -1].item() == stop_id:
                return True
        return False
