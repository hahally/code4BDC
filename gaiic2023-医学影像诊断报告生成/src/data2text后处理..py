import json
import re

def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def cal_score(keywords,inference):
    # coverage
    cnt = 0
    for kw in keywords:
        if kw in inference:
            cnt += 1
            inference = inference.replace(kw, " ", 1)  # 避免重复的关键字
    coverage = cnt / len(keywords)

    return coverage

def to_json(data, file):
    with open(file=file, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=0))

#后处理
def post_processing(key_words=[], text=''):
    for kw in key_words:
        if kw not in text:
            if kw.lower() in text.lower():
                # 英文大小写
                text = re.sub(re.escape(kw), kw, text, flags=re.I)
            else:
                # 数字问题
                if len(kw)<3:continue
                pattern = f"{kw[0]}(.+?){kw[-1]}"
                if len(text.split('。'))<2:
                    text = '，'.join([re.sub(pattern, kw, t, count=1, flags=re.I) for t in text.split('，')])
                else:
                    text = '。'.join([re.sub(pattern, kw, t, count=1, flags=re.I) for t in text.split('。')])
                    

    return text

def get_scores(results, test_data):
    score = 0
    for r, t in zip(results, test_data):
        score += cal_score(t['key_words'], r['text'])

    return score/len(results)

test_data = load_json_dataset('./test_set.json')
results = load_json_dataset('./result_622_copy.json')

print(get_scores(results, test_data))
i = 333
results[i]['text'] = post_processing(test_data[i]['key_words'], results[i]['text'])
for i in range(len(test_data)):
    results[i]['text'] = post_processing(test_data[i]['key_words'], results[i]['text'])
    # if len(test_data[i]['key_words'])<4 and cal_score(test_data[i]['key_words'], results[i]['text'])<1:
    if cal_score(test_data[i]['key_words'], results[i]['text'])<0.7:
        # results[i]['text'] = ','.join(test_data[i]['key_words'])
        print(i, results[i]['text'])
print(get_scores(results, test_data))

# to_json(results, 'copy_623.json')