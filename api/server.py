from flask import Flask, request, jsonify, abort, Response
from allennlp.predictors.predictor import Predictor
import scrape_schema_recipe

app = Flask(__name__)

@app.route("/", methods=["GET"])
def info():
    return 'This is the info route to the mr-cookie api.'

#answer question to a recipe with allennlp machine comprehension model
@app.route("/question", methods=["POST"])
def answer_question():
    print(request)
    recipe = request.json["recipe"]
    question = request.json["question"]
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    prediction = predictor.predict(passage=recipe, question=question)
    return prediction["best_span_str"]

#scrapes a recipe page for structured data of the type recipe and respond the structured recipe
@app.route("/recipe", methods=["GET"])
def get_recipe():
    url = request.args.get('url')
    recipe_list = scrape_schema_recipe.scrape_url(url)
    if len(recipe_list) < 1:
        abort(Response("Sorry couldn't find a recipe on this page."))
    else:
        return jsonify(recipe_list[0])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)