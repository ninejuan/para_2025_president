import os
import json
import boto3
from typing import List, Optional

class BedrockTextGenerator:
    def __init__(
        self, 
        model_id: str = 'ai21.jamba-1-5-mini-v1:0', 
        region_name: str = 'us-east-1',
        max_tokens: int = 200,
        temperature: float = 0.7
    ):
        """
        AWS Bedrock AI21 Jamba 텍스트 생성기 초기화
        
        Args:
            model_id (str): 사용할 Bedrock 모델 ID
            region_name (str): AWS 리전
            max_tokens (int): 최대 생성 토큰 수
            temperature (float): 생성 다양성 조절
        """
        # AWS 자격증명 환경변수 확인
        if not all(key in os.environ for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']):
            raise EnvironmentError("AWS 자격증명 환경변수를 설정해주세요.")

        # Bedrock 클라이언트 생성
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime', 
            region_name=region_name
        )

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 채팅 히스토리 저장
        self.chat_history = []

    def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        텍스트 생성 메서드
        
        Args:
            prompt (str): 사용자 입력 프롬프트
            system_prompt (str, optional): 시스템 프롬프트
        
        Returns:
            생성된 텍스트 응답
        """
        # 전체 프롬프트 구성
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n"
        
        # 이전 대화 히스토리 추가
        if self.chat_history:
            full_prompt += "대화 히스토리:\n"
            for history_item in self.chat_history:
                full_prompt += f"{history_item}\n"
        
        full_prompt += f"Human: {prompt}\nAssistant:"

        # 모델 입력 페이로드 구성 (AI21 Jamba 모델용)
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": self.temperature,
        }

        try:
            # Bedrock API 호출
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            generated_text = response_body['choices'][0]['message']['content']

            self.chat_history.append(f"Human: {prompt}")
            self.chat_history.append(f"Assistant: {generated_text}")

            return generated_text.strip()

        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            return ""

    def interactive_chat(self, system_prompt: Optional[str] = None):
        """
        대화형 인터페이스
        
        Args:
            system_prompt (str, optional): 시스템 프롬프트
        """
        print("AI 텍스트 생성기 시작. '종료'를 입력하면 종료합니다.")

        while True:
            user_input = input("사용자: ")
            
            if user_input == '종료':
                print("텍스트 생성기를 종료합니다.")
                break
            
            try:
                response = self.generate_text(
                    prompt=user_input, 
                    system_prompt=system_prompt
                )
                
                print(f"AI: {response}")
            
            except Exception as e:
                print(f"오류 발생: {e}")

# 메인 실행부
if __name__ == "__main__":
    # AWS 자격증명 환경변수 설정 예시
    os.environ['AWS_ACCESS_KEY_ID'] = ''
    os.environ['AWS_SECRET_ACCESS_KEY'] = ''
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

    text_generator = BedrockTextGenerator()
    text_generator.interactive_chat(
        system_prompt="너는 친절하고 도움이 되는 AI 어시스턴트야."
    )