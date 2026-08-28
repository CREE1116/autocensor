# AutoCensor

애니메이션/일러스트 이미지의 **남성기 · 여성기 · 유두/유륜**을 자동 검출해 흰색(또는 검정/모자이크/블러)으로 마스킹하는 Electron 앱. 폴더 단위 배치 처리 지원.

## 실행

```bash
npm install
npm start          # 개발 실행
npm run build      # dist/mac-arm64/AutoCensor.app 생성
```

## 검출 모델 (앙상블)

모델마다 학습 데이터가 달라서 **놓치는 게 서로 다르다.** 검열은 어차피 마스크의 합집합만 필요하므로, 여러 개를 돌려 OR하면 병합 로직 없이 재현율만 올라간다. 사이드바에서 체크박스로 조합한다.

| 키 | 출처 | task | 해상도 | 클래스 |
|---|---|---|---|---|
| `anime-medium` | `01miku/anime-nsfw-segm-yolo26` | seg | 1280px | anus, nipple, penis, vagina, female/male face, pubic hair |
| `anime-xl` | `01miku/anime-nsfw-segm-yolo26` | seg | 1280px | anus, nipple, penis, vagina, female/male face, pubic hair (초대형 고성능) |
| `anime-nano` | `01miku/anime-nsfw-segm-yolo26` | seg | 640px | anus, nipple, penis, vagina, female/male face, pubic hair (빠른 확인용) |

기본 추천 모델은 `anime-medium` (1280px 고해상도, 7개 클래스 지원). 최고 정밀도를 원할 경우 `anime-xl`을, 빠른 속도를 원할 경우 `anime-nano`를 선택하거나 조합할 수 있다.

모델 파일이 로컬에 없는 경우 사이드바에서 `[다운로드]` 또는 `[누락된 모델 다운로드]` 버튼으로 프로그램 내에서 즉시 자동 다운로드할 수 있다.

각 모델의 클래스는 `electron/models.js`의 `map`으로 공통 라벨(`electron/labels.js`)에 매핑된다. `map`에 없는 클래스는 무시된다. `.pt` 원본은 `tools/src-models/`에 위치하며 필요 시 파인튜닝 탭에서 자동 다운로드/선택 가능하다.

```bash
# .pt -> ONNX 재변환
python3 -c "from ultralytics import YOLO; YOLO('tools/src-models/ntd11_v5.pt').export(format='onnx', imgsz=640, opset=12)"
```

### 유륜 처리

어떤 모델에도 `areola` 클래스가 없고 `nipple` 윤곽은 유두 끝만 덮는다. 유륜은 **형태가 아니라 색 영역**이고 유두 대비 크기가 그림마다 달라서, 고정 배율은 작은 유륜에선 피부까지 덮고 큰 유륜에선 링이 남는다.

핵심은 **유륜이 유두와 동심원**이라는 것 — 경계가 각도의 함수 `r(θ)`이므로 2D 분할 문제가 1D 엣지 탐색 N개로 분해된다.

`electron/rayfit.js` (기본, `method: 'ray'`)

1. 유두 중심에서 180방향으로 광선을 쏜다. **광선마다 자기 끝부분에서 피부 기준색**을 뽑으므로 몸을 가로지르는 조명 그라데이션이 영향을 못 준다.
2. 각 광선에서 바깥으로 걸으며 "피부색과 다른" 마지막 반지름을 찾는다. 짧은 공백은 건너뛰어 유륜 위의 하이라이트나 선을 이어준다.
3. 180개 반지름에 **타원 `Ax² + Bxy + Cy² = 1`을 3파라미터 최소제곱으로 피팅**하고, 잔차가 큰 광선을 잘라내며 3회 재가중(IRLS)한다. 머리카락 한 가닥이 광선 몇 개를 늘려도 피팅은 거의 안 움직인다.
4. 타원이 퇴화하면 광선이 그린 별모양 폴리곤(원형 중앙값 필터)으로 폴백.

광선은 옆으로 새지 않으므로 그림자를 타고 번지는 문제가 구조적으로 없다. `electron/colorgrow.js`(`method: 'flood'`)는 4-이웃 BFS 확산 방식으로 남겨뒀다 — 유륜이 타원이 아닌 특이한 경우용.

실측 (`test/areola-hard.js`, 30° 회전 타원 + 조명 그라데이션 + 머리카락 + 하이라이트): 면적 오차 **1.3%**, 머리카락 따라가지 않음, 하이라이트 아래도 덮음.

사이드바 **유륜 찾기 방식**에서 ray/flood/끄기, 허용 오차, 최대 탐색 배율, 실패 시 고정 배율을 조절한다.

## 동작 구조

- `electron/preprocess.js` — ultralytics 방식 letterbox(중앙 정렬, gray 114). sharp가 한 파이프라인에서 resize/extend 순서를 뒤집으므로 단계마다 `toBuffer()`로 끊고 최종 raw 크기를 assert한다.
- `electron/detector.js` — 추론 → 클래스별 NMS → prototype mask 디코딩 → 원본 해상도 커버리지 마스크에 합성.
  - **타일링**: 긴 변이 모델 입력의 1.4배를 넘으면 20% 겹침 타일로 쪼개 원본 해상도로 각각 추론한다(고해상도 이미지에서 작은 부위가 뭉개지는 문제 해결). 타일에 걸친 큰 객체를 위해 전체 이미지 패스도 항상 같이 돌린다. 마스크는 합집합이라 중복 검출은 무해하다.
- `electron/censor.js` — 마스크 하나로 합친 뒤 효과를 이미지 전체에 **한 번만** 적용하고 마스크로 합성한다(검출별 처리는 겹친 곳이 두 번 뭉개지고 페더링 경계가 남는다).
  - **소프트 브러시 경계**: 원하는 경계보다 더 팽창시킨 뒤 같은 양만큼 블러한다. 그냥 블러만 하면 작은 영역은 가운데까지 반투명해진다. `edgeGamma`(<1)로 알파가 늦게 떨어지게 조절.
  - 슬라이더 값은 긴 변 1000px 기준이고 실제 해상도에 비례 적용된다(`scaleWithResolution`).
  - 모자이크 블록 = `max(4, ceil(긴 변 / 100))` (FANZA/DLsite 규정), 격자는 이미지 전체 기준.
- `electron/batch.js` — 폴더 순회(하위 폴더 포함), 이어하기(기존 결과 건너뛰기), 형식 변환, 취소.

## 검토 · 보정 · 데이터셋

배치가 끝나면 결과 폴더에 사이드카가 생긴다:

```
<출력폴더>/_autocensor/
  manifest.json                  # 원본→결과 매핑, 검출 목록, 라벨 목록
  masks/<key>__union.png         # 실제로 검열에 쓰인 합집합 마스크
  masks/<key>__nipple.png        # 라벨별 마스크 (보정 → 학습 라벨로 직행)
```

**검토 / 보정 탭** — 결과 폴더를 열면 그리드로 전부 보여준다. 필터(검열됨/검출 없음/보정됨), 썸네일 크기 조절. 셀을 클릭하면 편집기가 열린다.

**편집기** — 원본 위에 현재 마스크가 반투명 빨강으로 얹혀 나온다.

| 조작 | 키 |
|---|---|
| 칠하기 / 지우기 | `B` / `E` |
| 되돌리기 | `Cmd+Z` |
| 닫기 | `Esc` |

- 칠할 때 **부위**를 고르면 합집합 마스크와 그 부위 마스크에 동시에 들어간다. 지우개는 모든 부위에서 지운다 — 안 그러면 사용자가 방금 걷어낸 영역이 학습 라벨로 남는다.
- **적용 후 저장**: 보정된 마스크로 원본을 다시 검열해 결과 파일을 덮어쓰고, 사이드카 마스크와 매니페스트도 갱신한다. 검열 방식·부드러움 등은 사이드바의 현재 설정을 그대로 쓴다.
- **데이터셋에 추가**: 붓칠한 것이 곧 그 이미지가 필요로 했던 부위별 정답이므로, 그대로 파인튜닝 라벨이 된다.

### 데이터셋 형식

ultralytics YOLO 세그멘테이션 레이아웃이라 `yolo train` 이 폴더를 바로 먹는다.

```
<데이터셋>/
  data.yaml                      # names: 0..11 (labels.js의 공통 라벨)
  images/train/<이름>_<해시>.png
  labels/train/<이름>_<해시>.txt  # cls x1 y1 x2 y2 ... (정규화 폴리곤)
```

마스크 → 폴리곤 변환은 `electron/contour.js`: 8-연결 컴포넌트 → Moore 경계 추적 → Douglas-Peucker 단순화. 라벨 파일이 커지지 않게 점이 200개를 넘으면 epsilon을 키워 다시 단순화한다. 아무것도 안 칠한 이미지는 빈 `.txt`로 저장된다 — 유효한 네거티브 샘플이다.

이름에 원본 경로 해시가 붙어서 같은 원본을 다시 추가하면 덮어쓴다(중복 안 쌓임).

## 파인튜닝 탭

학습은 **사용자의 Python**에서 돌린다 — ultralytics가 torch를 끌고 오는데 앱에 번들하기엔 너무 크다. 앱은 설정을 넘기고, 출력을 스트리밍하고, 나온 모델을 등록하는 일만 한다.

```bash
pip install ultralytics          # 없으면 탭 상단에 안내가 뜬다
```
특정 파이썬을 쓰려면 `AUTOCENSOR_PYTHON=/경로/python3` 환경변수로 지정.

- **베이스**: 기본값은 `ntd11 v5 (권장, 애니 7클래스 기본)`. 그 외 `wenaka`, `nipples-seg`, `thatsca-nipples` 등 원래 쓰던 애니 NSFW 모델 가중치 또는 일반 YOLO(`yolo11n/s-seg`) 및 로컬 `.pt` 선택 가능.
- epochs / imgsz / batch / device(Apple Silicon은 `mps`) 조절
- 데이터셋 20장 미만이면 과적합 경고를 띄운다
- 로그 실시간 출력, 중지 버튼

학습이 끝나면 `best.pt`를 ONNX로 내보내고 **앱을 재시작하지 않고** 사이드바 모델 목록에 `학습됨` 배지로 나타난다. 저장 위치는 `~/Library/Application Support/autocensor/models/`(`.onnx` + 같은 이름의 `.json` 서술자) — 앱 번들 밖이라 재설치해도 남는다. 학습 산출물은 `.../autocensor/runs/`.

서술자는 `electron/models.js`의 항목과 같은 형식이라, 손으로 `.onnx` + `.json`을 넣어도 그대로 모델이 하나 추가된다.

터미널에서 직접 돌리고 싶으면:
```bash
yolo segment train data=<데이터셋>/data.yaml model=yolo11n-seg.pt imgsz=640 epochs=100 device=mps
```

## 렌더러 스크립트 주의

`renderer/*.js`는 classic script라 최상위 `function foo()`가 곧 `window.foo`다. 다른 렌더러 파일에 함수를 내보낼 때 `window.foo = () => foo()`로 쓰면 **그 전역을 래퍼로 덮어써서 래퍼가 자기 자신을 호출** — `Maximum call stack size exceeded`가 난다. 반드시 다른 이름으로 참조 대입할 것:

```js
function currentCensorOptions() { ... }
window.censorOptions = currentCensorOptions;   // 참조, 섀도잉 불가
```

`test/renderer-globals.js`가 이 패턴을 정적으로 잡고, 스텁 DOM 위에서 renderer.js를 실제 로드해 호출까지 확인한다.

## UI

- 탭 4개: **배치 처리 / 미리보기 / 검토·보정 / 파인튜닝**
- **배치 처리** 탭: 입력/출력 폴더 지정 → 시작. 파일별 검출 결과가 로그에 뜬다. `마스크 저장`을 끄면 검토 탭을 못 쓴다.
- **미리보기** 탭: 이미지 1장으로 임계값·확장·마스크 모양을 조정해 본 뒤 배치에 적용.
- 사이드바: 모델, 타일링, 검열 방식/모양, 여유 확장(dilate), 경계 페더, 부위별 임계값·확장 배율.

## 테스트

```bash
node test/coord-roundtrip.js   # 조작한 모델 출력으로 좌표 변환 왕복 검증
node test/mosaic-check.js      # 모자이크가 실제로 걸리는지 검증
node test/smoke.js             # 실제 모델 추론 + 4개 검열 모드
node test/contour-expand.js    # expand가 박스가 아니라 윤곽을 키우는지 검증
node test/areola-grow.js       # 유륜 크기를 색으로 따라가는지 + 가드 검증
node test/areola-hard.js       # 회전 타원 + 조명 + 머리카락 + 하이라이트
node test/ensemble-smoke.js    # 모델별 속도 + 앙상블
node test/contour.js           # 마스크 → 폴리곤 면적/점 개수
node test/dataset-roundtrip.js # 마스크 → YOLO 라벨 → 래스터화 IoU (0.97+)
node test/review-flow.js       # 배치 → 매니페스트 → 붓칠 → 재검열 → 데이터셋
node test/large-tree.js        # 20만 파일 트리 순회 (spread 스택 오버플로 회귀)
node test/renderer-globals.js  # classic script 전역 섀도잉 재귀 회귀
node test/train-e2e.js         # 실제 학습 → ONNX → 모델 등록 → 추론 (python 필요)
node test/soft-edge.js         # 코어는 불투명, 바깥으로 단조 감소하는 알파 램프인지 검증
node test/download-test.js     # 모델 다운로더, 리다이렉트 추적, 진행률 콜백 검증
node test/visual-compare.js    # test/visual-compare.png 생성 (원본 | 고정배율 | 색상추적)
```
