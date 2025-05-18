from flask import Flask, request, jsonify
import os
from crew import CrewQAFlow, CrewQAState
from custom_types import UserPreferences
 
app = Flask(__name__)


@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    context = data.get('context', '')
    user_preferences = data.get('user_preferences', {
        'answer_length': 'medium',
        'expertise_level': 'intermediate'
    })
    print(question, context)
       
    flow = CrewQAFlow()
    answer = flow.kickoff(inputs = {
        "input_query": question,
        "context": context,
        "user_preferences": UserPreferences(**user_preferences),
        "mode": "qa"
    })
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return jsonify({"answer": answer}), 200

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.json  
    print("DATAAA   ")  
    print(data)
    if data is None:
        print("NO DATA PROVIDED")
        data = {}
    
    print("USER PREFERENCES")
    user_preferences = data.get('user_preferences', {
        'answer_length': 'medium',
        'expertise_level': 'intermediate'
    })
    print("USER PREFERENCES")
    flow = CrewQAFlow()
    print("FLOW")
    answer = flow.kickoff(inputs = {
        "context": data.get('context', ''),
        "user_preferences": UserPreferences(**user_preferences),
        "mode": "summarize"
    })
    print("================================================")
    return jsonify({"summary": answer}), 200

@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    data = request.json
    if not data or 'claim' not in data or data['claim'] == "":
        return jsonify({"error": "Statement to fact-check is required"}), 400
    
    statement = data['claim']
    context = data.get('context', '')
    
    flow = CrewQAFlow()
    answer = flow.kickoff(inputs = {
        "claim_to_verify": statement,
        "context": context,
        "mode": "fact_check"
    })
    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    return jsonify({"fact_check_result": answer}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
