# # 定义训练时评价指标
#
# 仅供训练时参考, 包含实体的precision，recall以及f1。
#
# 只有和标注的数据完全相同才算是1，否则为0

# In[ ]:


# 训练时指标
class EvaluateScores:
    def __init__(self, entities_json, predict_entities_json):
        self.entities_json = entities_json
        self.predict_entities_json = predict_entities_json

    def compute_entities_score(self):
        return evaluate_entities(self.entities_json, self.predict_entities_json, list(set(self.entities_json.keys())))


def evaluate_entities(true_entities, pred_entities, entity_types):
    scores = []

    ps2 = []
    rs2 = []
    fs2 = []

    for ent_type in entity_types:
        true_entities_list = true_entities.get(ent_type, [])
        pred_entities_list = pred_entities.get(ent_type, [])
        s = _compute_metrics(true_entities_list, pred_entities_list)
        scores.append(s)
    ps = [i['p'] for i in scores]
    rs = [i['r'] for i in scores]
    fs = [i['f'] for i in scores]
    s = {
        'p': sum(ps) / len(ps),
        'r': sum(rs) / len(rs),
        'f': sum(fs) / len(fs),
    }
    return s


def _compute_metrics(ytrue, ypred):
    ytrue = set(ytrue)
    ypred = set(ypred)
    tr = len(ytrue)
    pr = len(ypred)
    hit = len(ypred.intersection(ytrue))
    p = hit / pr if pr!=0 else 0
    r = hit / tr if tr!=0 else 0
    f1 = 2 * p * r / (p + r) if (p+r)!=0 else 0
    return {
        'p': p,
        'r': r,
        'f': f1,
    }