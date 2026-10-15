# 최종 실행 설정 기록

두 *.run_config.json은 ZIP의 최종 run_config에서 경로만 상대 경로로 정리한 기록입니다.
training CLI는 이 JSON을 직접 읽지 않습니다. 실행 옵션은 WORKFLOW.md를 참고하세요.
학습을 그대로 재현하는 완전한 환경 lock/중간 adapter 기록은 아닙니다.

- nikl_v4.run_config.json: NIKL v4 최종 추가 학습 설정
- gksl_v1.run_config.json: GKSL 최종 설정

실제 어댑터 폴더에는 별도 run_config.json과 tokenizer/weights가 필요합니다.
scripts.restore_local_assets가 ZIP에서 복원하면서 현재 경로로 변환합니다.
