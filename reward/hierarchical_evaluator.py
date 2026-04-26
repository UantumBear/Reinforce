import re

class HierarchicalEvaluator:
    """
    - improve 모드에서 사용되는 main Evaluator 입니다.
    - 계층적 피드백을 제어하는 메인 Evaluator 입니다.
    
    두개의 레이어로 구성합니다.
    Layer 1: 정답 여부 판단 (Hard Gate)
    Layer 2: 스타일/형식 판단
    Layer 3: 사용자 정의 판단 (헌법)

    가장 중요한 정답 여부를 먼저 판단하고, 
    정답이 틀릴 경우 evaluation_instruction 을 구성하는 문자열이
    논리적 오류에 집중하고, 쓸데없는 노이즈를 생성하지 않도록 제어합니다.

    정답이 맞는 경우에는, 논리적 오류에 대한 피드백은 생성하지 않고,
    스타일/형식에 대한 피드백이 evaluation_instruction 내에 포함되도록 제어합니다.

    스타일/형식 판단은 누구나 주관적일 수 있습니다. 때문에, 
    sementic similarity 계산과 같은 객관적인 지표를 활용하여,
    스타일/형식이 얼마나 모법답안과 유사한지 llm 이 객관적으로 판단할 수 있도록 수치를 제공합니다.

    Layer 3에서는 사용자가 특별히 무조건 지켰으면 하는, 피드백이 있다면
    해당 부분을 만족하는지 판단하고 피드백을 생성하도록 제어합니다. 
    이는 '헌법' 과 같은 역할을 하나, 3순위로 배정하였습니다.
    그 이유는 일단 정답을 맞출 수 있도록 프롬프트 최적화를 진행하며,
    사용자 요구사항 '헌법' 은 False/True 로 결과값을 제공하여,
    0.5 -> 0.6 -> 0.7 -> 0.8 -> 0.9 -> 1.0 으로 점진적으로 개선해 나가는 흐름은 유지하되,
    사용자가 해당 프롬프트를 채택하지 않을 수 있도록 하는 방향으로 개선하기 위함입니다. 


    """
    def __init__(self, judge_llm):
        self.judge_llm = judge_llm

    @staticmethod
    def _extract_first_tag_content(text: str, tag_name: str) -> str | None:
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def validate_feedback_output_tags(self, feedback_text: str) -> dict:
        """
        improve 모드에서만 사용한다.
        개선 모드 피드백 문자열에 필수 출력 태그 (improve_objective_function 형식)
        가 모두 존재하는지 검사한다.

        """
        required_tag_pairs = [
            ("<accuracy_analysis>", "</accuracy_analysis>"),
            ("<accuracy>", "</accuracy>"),
            ("<style_format_analysis>", "</style_format_analysis>"),
            ("<similarity_score>", "</similarity_score>"),
            ("<total_analysis>", "</total_analysis>"),
        ]

        text = "" if feedback_text is None else str(feedback_text)

        missing_open_tags = []
        missing_close_tags = []
        missing_pairs = []
        nested_errors = []

        # 첫 루트 태그(<accuracy_analysis>) 이전에 텍스트가 있으면 경고
        first_root_idx = text.lower().find("<accuracy_analysis>")
        has_prefix_before_iteration = False
        if first_root_idx > 0 and text[:first_root_idx].strip():
            has_prefix_before_iteration = True
            nested_errors.append("<accuracy_analysis> 태그 이전에 불필요한 텍스트가 존재합니다.")

        for open_tag, close_tag in required_tag_pairs:
            has_open = open_tag in text
            has_close = close_tag in text

            if not has_open:
                missing_open_tags.append(open_tag)
            if not has_close:
                missing_close_tags.append(close_tag)
            if not (has_open and has_close):
                missing_pairs.append(f"{open_tag} ... {close_tag}")

        # 중첩 구조 검사 1: accuracy_analysis 내부에 accuracy + review
        accuracy_analysis_content = self._extract_first_tag_content(text, "accuracy_analysis")
        if accuracy_analysis_content is not None:
            lower_block = accuracy_analysis_content.lower()
            if "<accuracy>" not in lower_block or "</accuracy>" not in lower_block:
                nested_errors.append("<accuracy> 태그 쌍이 <accuracy_analysis> 내부에 없습니다.")
            if "<review>" not in lower_block or "</review>" not in lower_block:
                nested_errors.append("<review> 태그 쌍이 <accuracy_analysis> 내부에 없습니다.")
        elif "<accuracy_analysis> ... </accuracy_analysis>" not in missing_pairs:
            nested_errors.append("<accuracy_analysis> 태그 내부 구조를 확인할 수 없습니다.")

        # 중첩 구조 검사 2: style_format_analysis 내부에 similarity_score + review
        style_format_content = self._extract_first_tag_content(text, "style_format_analysis")
        if style_format_content is not None:
            lower_block = style_format_content.lower()
            if "<similarity_score>" not in lower_block or "</similarity_score>" not in lower_block:
                nested_errors.append("<similarity_score> 태그 쌍이 <style_format_analysis> 내부에 없습니다.")
            if "<review>" not in lower_block or "</review>" not in lower_block:
                nested_errors.append("<review> 태그 쌍이 <style_format_analysis> 내부에 없습니다.")
        elif "<style_format_analysis> ... </style_format_analysis>" not in missing_pairs:
            nested_errors.append("<style_format_analysis> 태그 내부 구조를 확인할 수 없습니다.")

        # 중첩 구조 검사 3: total_analysis 내부에 review
        total_analysis_content = self._extract_first_tag_content(text, "total_analysis")
        if total_analysis_content is not None:
            lower_block = total_analysis_content.lower()
            if "<review>" not in lower_block or "</review>" not in lower_block:
                nested_errors.append("<review> 태그 쌍이 <total_analysis> 내부에 없습니다.")
        elif "<total_analysis> ... </total_analysis>" not in missing_pairs:
            nested_errors.append("<total_analysis> 태그 내부 구조를 확인할 수 없습니다.")

        return {
            "is_valid": len(missing_pairs) == 0 and len(nested_errors) == 0,
            "missing_open_tags": missing_open_tags,
            "missing_close_tags": missing_close_tags,
            "missing_pairs": missing_pairs,
            "nested_errors": nested_errors,
            "has_prefix_before_iteration": has_prefix_before_iteration,
        }

    def fix_feedback_output_tags(self, feedback_text: str) -> str:
        """
        improve 모드 피드백의 태그 구조를 파이썬 코드로 보정한다.

          보정 규칙:
          1) <accuracy_analysis> 앞에 텍스트가 있으면 제거
        2) 필수 태그가 없으면 "un generated" 기본값으로 생성
          3) <accuracy> + <review> 를 <accuracy_analysis> 내부에 강제 배치
          4) <similarity_score> + <review> 를 <style_format_analysis> 내부에 강제 배치
          5) <review> 를 <total_analysis> 내부에 강제 배치

        @param feedback_text: Judge 결과 문자열
        @return: 보정된 문자열
        """
        source_text = "" if feedback_text is None else str(feedback_text)

        # <accuracy_analysis> 이전 텍스트 제거
        lower_source = source_text.lower()
        root_start_idx = lower_source.find("<accuracy_analysis>")
        if root_start_idx >= 0:
            working_text = source_text[root_start_idx:]
        else:
            working_text = source_text

        def _value_or_default(tag_name: str, text_block: str) -> str:
            extracted = self._extract_first_tag_content(text_block, tag_name)
            if extracted is None or not extracted.strip():
                return "un generated"
            return extracted.strip()

        # 각 루트 블록 내부 우선 추출
        accuracy_analysis_content = self._extract_first_tag_content(working_text, "accuracy_analysis")
        style_format_content = self._extract_first_tag_content(working_text, "style_format_analysis")
        total_analysis_content = self._extract_first_tag_content(working_text, "total_analysis")

        accuracy_source = accuracy_analysis_content if accuracy_analysis_content is not None else working_text
        style_source = style_format_content if style_format_content is not None else working_text
        total_source = total_analysis_content if total_analysis_content is not None else working_text

        accuracy_value = _value_or_default("accuracy", accuracy_source)
        similarity_score_value = _value_or_default("similarity_score", style_source)

        # review 태그는 블록별로 개별 추출
        accuracy_review_value = _value_or_default("review", accuracy_source)
        style_review_value = _value_or_default("review", style_source)
        total_review_value = _value_or_default("review", total_source)

        return (
            "<accuracy_analysis>\n"
            f"    <accuracy> {accuracy_value} </accuracy>\n"
            f"    <review> {accuracy_review_value} </review>\n"
            "</accuracy_analysis>\n"
            "\n"
            "<style_format_analysis>\n"
            f"    <similarity_score> {similarity_score_value} </similarity_score>\n"
            f"    <review> {style_review_value} </review>\n"
            "</style_format_analysis>\n"
            "\n"
            "<total_analysis>\n"
            f"    <review> {total_review_value} </review>\n"
            "</total_analysis>"
        )

    # 아직 설계 중
    # def __call__(self, prediction_var: tg.Variable, ground_truth: str) -> tg.Variable:
    #     prediction_text = prediction_var.value
        
    #     # 1. 파이썬 코드로 1차 채점 (Hard Gate)
    #     is_correct = string_based_equality_fn(prediction_text, ground_truth)
        
    #     # ----------------------------------------------------
    #     # Layer 1: 오답인 경우 -> 논리 오류만 피드백
    #     # ----------------------------------------------------
    #     if not is_correct:
    #         prompt = get_logic_diagnosis_prompt(ground_truth, prediction_text)
    #         feedback_text = self.judge_llm(prompt) # JudgeLLM 호출
            
    #         loss_var = tg.Variable(value="0", role_description="Accuracy Score")
    #         loss_var.set_grad_text(feedback_text)
    #         return loss_var
            
    #     # ----------------------------------------------------
    #     # Layer 2: 정답인 경우 -> 스타일/형식 피드백 (조건부)
    #     # ----------------------------------------------------
    #     else:
    #         similarity = calculate_similarity(prediction_text, ground_truth)
            
    #         # 정답은 맞췄는데, 스타일이 기준 미달일 때만
    #         if similarity < 0.9: 
    #             prompt = get_style_feedback_prompt(ground_truth, similarity)
    #             feedback_text = self.judge_llm(prompt) # JudgeLLM 호출
                
    #             loss_var = tg.Variable(value="0.5", role_description="Penalty for Style")
    #             loss_var.set_grad_text(feedback_text)
    #             return loss_var
                
    #         # 정답도 맞추고 스타일도 완벽할 때 -> 노이즈 차단!
    #         else:
    #             loss_var = tg.Variable(value="1", role_description="Perfect Score")
    #             loss_var.set_grad_text("") # 피드백 텍스트(Gradient)를 비워버림
    #             return loss_var