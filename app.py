from pathlib import Path

from flask import Flask, render_template_string, request
import os

from inference import load_predictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100_000
root = Path(__file__).parent
backend = os.getenv("NOONCHI_MODEL", "baseline")
default_path = "koelectra_results/final" if backend == "koelectra" else "results/baseline.joblib"
predict_scores = load_predictor(backend, os.getenv("NOONCHI_MODEL_PATH") or root / default_path)


def predict(context, response):
    score = float(predict_scores([context], [response])[0])
    return int(score > 0.5), score
notice = "AI의 예측은 상대방의 진짜 속마음을 확정하는 것이 아닙니다. 비꼼으로 보이는 표현도 실제로는 다른 의도일 수 있습니다. 추측보다 직접적인 확인과 대화를 권장합니다."


@app.errorhandler(413)
def too_large(error):
    return render_template_string(PAGE, context="", response="", situation="personal",
                                  result=None, error="입력 데이터가 너무 큽니다. 이전 대화는 4,000자, 마지막 발언은 1,000자 이내로 입력해 주세요.",
                                  notice=notice, model_name="KoELECTRA" if backend == "koelectra" else "TF-IDF + Logistic Regression"), 413


@app.route("/", methods=["GET", "POST"])
def home():
    context = request.form.get("context", "")
    response = request.form.get("response", "")
    situation = request.form.get("situation", "personal")
    result, error = None, None
    if request.method == "POST":
        if not response.strip() or len(context) > 4000 or len(response) > 1000:
            error = "마지막 발언을 입력하고 글자 수 제한을 확인해 주세요."
        else:
            prediction, score = predict(context, response)
            label = "비꼼 표현 가능성 높음" if prediction == 1 else "비꼼 표현 가능성 낮음"
            reply = ("제가 이해한 내용이 맞는지 확인하고 싶어요. 어떤 뜻인지 조금 더 설명해 주실 수 있을까요?"
                     if situation == "work" else "내가 다르게 받아들였을 수도 있어. 어떤 뜻이었는지 말해줄래?")
            followup = ("텍스트로는 뉘앙스가 헷갈렸어. 잠깐 이야기할 수 있을까?"
                        if abs(score - .5) < .15 else "내가 이해한 게 맞는지 확인하고 싶어. 한 번만 더 이야기해 줄래?")
            result = label, round(score * 100, 1), reply, followup
    return render_template_string(PAGE, context=context, response=response, situation=situation,
                                  result=result, error=error, notice=notice,
                                  model_name="KoELECTRA" if backend == "koelectra" else "TF-IDF + Logistic Regression"), 400 if error else 200


PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>눈치코치</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f3;color:#18392f;font:16px/1.7 "Malgun Gothic",system-ui,sans-serif}
main{width:min(1040px,calc(100% - 40px));margin:48px auto}h1{font-size:clamp(32px,6vw,48px);letter-spacing:-2px;margin:0}
header p{margin:8px 0}.layout{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;margin-top:28px}
form,section{background:white;padding:28px;border:1px solid #d5e2db;border-radius:20px;box-shadow:0 8px 30px #173b2510}
h2{font-size:22px;margin:0 0 14px}h3{font-size:17px;margin-top:26px}
label{display:block;font-weight:650;margin:18px 0 6px}textarea,select,button{font:inherit}
textarea,select{width:100%;padding:12px;border:1px solid #8aa397;border-radius:10px;background:#fbfdfb;color:inherit}
textarea{resize:vertical}button{width:100%;margin-top:24px;background:#185b43;color:white;border:0;border-radius:10px;padding:13px 22px;font-weight:700;cursor:pointer}
button:hover{background:#104733}button:focus-visible,textarea:focus-visible,select:focus-visible{outline:3px solid #6bab97;outline-offset:3px}
meter{display:block;width:100%;height:26px;accent-color:#237957}.error{padding:12px;background:#fff0ed;color:#962e20;border-radius:8px}
small,.muted{color:#526b60;font-size:14px}.score{font-size:36px;font-weight:750;margin:8px 0}
.uncertain{background:#fff7e4;padding:12px;border-radius:10px;color:#725313;font-size:14px}
li{margin:12px 0}.notice{margin-top:24px;border-left:3px solid #8caa9c;padding-left:16px;font-size:14px;color:#526b60}
@media(max-width:700px){main{margin:24px auto;width:calc(100% - 24px)}.layout{grid-template-columns:1fr;gap:16px}form,section{padding:20px}h1{letter-spacing:-1px}}
</style><main><header><h1>눈치코치</h1><p>판단보다 이해, 추측보다 확인.</p>
<p>한국어 대화의 마지막 발언을 분석하고 오해를 줄이는 답변을 제안합니다.</p>
 </header><div class="layout"><form method="post"><h2>대화 입력</h2><label for="context">이전 대화</label>
<textarea id="context" name="context" rows="4" maxlength="4000">{{ context }}</textarea>
<label for="response">분석할 마지막 발언</label>
<textarea id="response" name="response" rows="2" maxlength="1000" required>{{ response }}</textarea>
<label for="situation">대화 상황</label><select id="situation" name="situation">
<option value="personal">친구·가족</option><option value="work" {% if situation == 'work' %}selected{% endif %}>직장·공식 대화</option>
</select><button>분석</button>
{% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}</form>
<section aria-live="polite" aria-atomic="true"><h2>분석 결과</h2><p class="muted">사용 모델 · {{ model_name }}</p>
{% if result %}<h3>{{ result[0] }}</h3>
<p class="score">{{ result[1] }} <small>/ 100점</small></p>
<label for="score">모델의 비꼼 예측 점수</label>
<meter id="score" min="0" max="100" value="{{ result[1] }}">{{ result[1] }}</meter>
<p class="muted">상대방의 실제 감정이나 의도를 확정하는 확률로 해석할 수 없습니다.</p>
{% if 40 <= result[1] <= 60 %}<p class="uncertain">40~60점은 해석에 주의를 기울이기 위한 참고용 불확실 구간입니다. 검증된 통계적 임계값은 아닙니다.</p>{% endif %}
<h3>이렇게 확인해 보세요</h3><ul><li>{{ result[2] }}</li><li>{{ result[3] }}</li></ul>
{% else %}<p class="muted">대화를 입력하고 분석을 누르면 예측 점수와 확인형 답변을 볼 수 있습니다.</p>{% endif %}
<p class="notice">{{ notice }}</p></section></div></main></html>"""


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "7860")))
