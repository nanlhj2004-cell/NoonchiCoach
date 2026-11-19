# 눈치코치 Colab 실행 안내

1. `NoonchiCoach_Complete_Colab.ipynb`를 열고 런타임에서 T4 GPU를 선택합니다. 맨 위부터 12단계를 실행합니다. 첫 셀에 실행 소스가 포함되어 있어 프로젝트 ZIP이나 별도 데이터 업로드 없이 시작할 수 있습니다.
2. 패키지 설치 셀은 pandas 2.2.3과 PyArrow 19 이상 24 미만을 사용합니다. Colab 기본 CUDA PyTorch 2.6 이상을 사용합니다. 패키지를 이미 불러온 세션이라면 세션을 재시작하고 처음부터 실행합니다.
3. Hugging Face에서 두 데이터셋을 받습니다. KoCoSa의 공식 분할로만 학습하고 원문 설명 필드는 제외합니다. NFC와 공백을 정규화하고 필수 결측과 정규화 입력 중복을 검사합니다. 라벨이 충돌하는 입력은 제거하고, 같은 라벨의 중복은 Test, Validation, Train 우선순위로 한 건을 보존합니다. 별도로 집계한 동일 맥락·응답과 화자 표기 통일 결과도 감사 JSON에서 확인할 수 있습니다. 의미상 재서술은 검사 범위 밖입니다.
4. Kocasm의 `0=no_sarcasm`, `1=sarcasm`은 공개 ClassLabel 정의를 따르며 런타임에서도 검증합니다. KoCoSa의 `Non-Sarcasm=0`, `Sarcasm=1`과 이진 분류 의미를 맞춥니다. Kocasm은 문맥 없는 소셜 문장입니다. 주석 기준과 입력 도메인이 다를 수 있으므로 성능 차이를 맥락 차이만으로 설명할 수 없습니다. Kocasm Train은 품질 검사에만 사용합니다. 두 모델 모두 동일한 정제 Kocasm Test와 빈 맥락으로 평가합니다. 충돌·중복 또는 KoCoSa 학습·검증 텍스트와 동일한 문장을 제외한 뒤 평가 건수를 기록합니다.
5. KoELECTRA는 학습률 2e-5, 3 epoch, seed 42, T4 FP16, 배치 8과 누적 2를 사용합니다. CUDA 메모리가 부족하면 새 학습 경로에서 배치 4 또는 2로 실행합니다. 선택 기준은 KoCoSa Validation Macro F1이며, KoCoSa Test와 Kocasm Test는 평가용입니다. 맥락형 베이스라인을 주 비교 대상으로 사용하고 마지막 발언형은 보조 실험으로 남깁니다.
6. 실행별 결과는 `/content/noonchi_complete/results/<실행 ID>/`에 남습니다. 원본과 전처리 데이터, 모델 다운로드 캐시 및 중간 체크포인트는 임시 디스크에 둡니다. 마지막 셀에서 `SAVE_TO_DRIVE = True`를 설정하면 최종 가중치와 평가 결과를 개인 Drive에 복사하고 저장량을 출력합니다. Drive 마운트에서 제공하는 여유 용량과 4GB 상한을 확인하며, 계정의 실제 남은 할당량도 확인하세요.

## Flask 연결

기존 베이스라인은 기본 경로에서 계속 실행됩니다. 새 실험을 연결하려면 `NOONCHI_MODEL=baseline`과 `NOONCHI_MODEL_PATH=<실행 폴더>/baseline/baseline.joblib`을 지정합니다. KoELECTRA는 `NOONCHI_MODEL=koelectra`와 `NOONCHI_MODEL_PATH=<실행 폴더>/koelectra/final`을 지정하고 상위 폴더의 `token_lengths.json`도 함께 둡니다. KoELECTRA에 필요한 PyTorch와 Transformers를 추론 환경에 설치합니다. 지원되는 입력 형식과 0.5 판정 경계는 노트북과 Flask에 공통입니다.

## 출처와 공개 조건

- [KoCoSa 데이터 카드](https://huggingface.co/datasets/YuminKim/KoCoSa)와 [논문](https://aclanthology.org/2024.lrec-main.864/)은 대화 맥락과 마지막 발언의 비꼼 여부를 다룹니다. 프로젝트에서 KoCoSa 재배포 및 파인튜닝 결과의 공개 조건을 확인하지 못했습니다.
- [Kocasm 데이터 카드](https://huggingface.co/datasets/SpellOnYou/kor_sarcasm)는 MIT로 표시하고, 라벨 이름을 `no_sarcasm`과 `sarcasm`으로 정의합니다. 원문은 실험 결과에 복사하지 않습니다. 오분류 CSV의 행 번호와 특성으로 원본을 비공개 상태에서 확인할 수 있습니다.
- [KoELECTRA 모델 카드](https://huggingface.co/monologg/koelectra-base-v3-discriminator)는 Apache-2.0으로 표시합니다.
