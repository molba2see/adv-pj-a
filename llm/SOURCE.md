# 코드 및 자료 출처

- 기반 코드: [molba2see/adv-pj-a의 기존 llm](https://github.com/molba2see/adv-pj-a/tree/19a8497dbcec496aec4f1f7c4bf1b429c79c9454/llm)
- 기반 commit: `19a8497dbcec496aec4f1f7c4bf1b429c79c9454`
- 최종 구현: 로컬 `llm최종.zip` (2026-08-30 패키지)
- 모델: [paust/pko-t5-base](https://huggingface.co/paust/pko-t5-base)
- 초기 비교: [AIRC-KETI/ke-t5](https://github.com/AIRC-KETI/ke-t5)
- NIKL 데이터: [모두의 말뭉치](https://kli.korean.go.kr/corpus/)

이번 코드 배포본은 모듈을 core/training/inference/data_processing/evaluation/tests로
정리하고 서버 절대 경로 이식, 선택적 로컬 자원 복원, CPU 회귀 테스트 환경을 추가했습니다.
실험의 학습 방식·라우팅·집계 수치는 최종 ZIP에 근거합니다.
원본/가공 corpus, 문법 원문 청크와 모델 가중치는 Git에 싣지 않습니다.

GKSL3k 원 배포 URL, 재난안전 배포 ID, 문법서 판본·서지정보는 기존 archive에서
확정할 수 없어 DATA.md에 명시했습니다. 모델과 데이터의 권리는 원 제공자에게 있으며
이 저장소의 공개가 외부 자료의 재배포 권한을 의미하지 않습니다.
원본 기반 코드에는 명시된 LICENSE 파일이 없어 이번 정리에서 임의 라이선스를 부여하지 않았습니다.
