import json
import matplotlib
from transformers import AutoModelForMaskedLM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

keys = ["sgns", "char", "electra"]
langs = ["en", "fr", "ru"]


plt.figure(dpi=1200)
EXP_NAME1='100_model_100_data'
MODEL_CAT1='top5'
epoch1='131'
EXP_NAME2='new_control_100_model_100_data'
MODEL_CAT2='Books'
epoch2='181'
model_path1 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}-mlm/epoch{epoch1}'
model_path2=f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}-mlm/epoch{epoch2}'
model1 = AutoModelForMaskedLM.from_pretrained(model_path1)
embeddings_1 = model1.bert.embeddings.word_embeddings.weight.data
embedding1_numpy = np.array(embeddings_1)
model2 = AutoModelForMaskedLM.from_pretrained(model_path2)
embeddings_2 = model2.bert.embeddings.word_embeddings.weight.data
embedding2_numpy = np.array(embeddings_2)
print(embedding1_numpy.shape)
files = [embedding1_numpy, embedding2_numpy]

X = []
for file in files:
    X += file
X_3d = PCA(n_components=2).fit_transform(X)
print(X_3d.shape)
data = {}
data["general"] = X_3d[:200]
data["control"] = X_3d[200:]

fig = plt.figure()
ax = fig.add_subplot(11)
for label, marker, color in zip(['general', 'control'], ['3', (5,2)], ["blue", "red"]):
    X_temp = data[label]
    ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
               label=label,
               marker=marker,
               color=color,
               alpha=0.5)
# if key == "char" and lang == "fr":
#     legend = ax.legend()
#     h, l = ax.get_legend_handles_labels()
#     l = [l[0], l[2], l[1]]
#     h = [h[0], h[2], h[1]]
#     legend = ax.legend(h,
#                        l,
#                        loc=9,
#                        fontsize=17.5,
#                        framealpha=0.6,
#                        markerscale=2)
#     for lh in legend.legendHandles:
#         lh.set_alpha(1)
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.axis('off')
# fig.savefig(lang + "." + key + ".trial.dist.pdf",
#             format='pdf',
#             bbox_inches='tight',
#             dpi=1200,
#             transparent=True)

plt.clf()