"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

@contextmanager
def supress_stdout_stderr():
	"""A context manager that redirects stdout and stderr to devnull"""
	with open(os.devnull, 'w') as fnull:
		with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
			yield (err, out)

def load_dataset(path):
	"""
	Carrega el dataset de registres dels ciclistes

	arguments:
		path -- dataset

	Returns: dataframe
	"""

	return pd.read_csv(path, delimiter=';')

def EDA(df):
	"""
	Exploratory Data Analysis del dataframe

	arguments:
		df -- dataframe

	Returns: None
	"""
	logging.debug('\n%s', df.shape)
	logging.debug('\n%s', df[:5])
	logging.debug('\n%s', df.columns)
	logging.debug('\n%s', df.info())

def clean(df):
	"""
	Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

	arguments:
		df -- dataframe

	Returns: dataframe
	"""
	df = df.drop('id', axis=1)
	df = df.drop('temps_total', axis=1)
	logging.debug('\nDataframe:\n%s\n...', df[:3])

	return df

def extract_true_labels(df):
	"""
	Guardem les etiquetes dels ciclistes (BEBB, ...)

	arguments:
		df -- dataframe

	Returns: numpy ndarray (true labels)
	"""
	df_true_labels = df.groupby(['tipus']).mean()
	logging.debug('\nDades agrupades per tipus:\n%s\n...', df_true_labels)
	true_labels = df["tipus"].to_numpy()

	return true_labels

def visualitzar_pairplot(df):
	"""
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	arguments:
		df -- dataframe

	Returns: None
	"""
	sns.pairplot(df)
	try:
		os.makedirs(os.path.dirname('img/'))
	except FileExistsError:
		pass
	plt.savefig("img/olav_martos_pairplot.png")

def clustering_kmeans(data, n_clusters=4):
	"""
	Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	arguments:
		data -- les dades: tp i tb

	Returns: model (objecte KMeans)
	"""
	model = KMeans(n_clusters=n_clusters, random_state=42)

	with supress_stdout_stderr():
		model.fit(data)

	return model

def visualitzar_clusters(data, labels):
	"""
	Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

	arguments:
		data -- el dataset sobre el qual hem entrenat
		labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

	Returns: None
	"""
	try:
		os.makedirs(os.path.dirname('img/'))
	except FileExistsError:
		pass
	fig = plt.figure()
	sns.scatterplot(x='temps_pujada', y='temps_baixada', data=data, hue=labels, palette="rainbow")
	plt.savefig("img/olav_martos_grafica1.png")
	fig.clf()
	#plt.show()

def associar_clusters_patrons(tipus, model):
	"""
	Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
	S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

	arguments:
	tipus -- un array de tipus de patrons que volem actualitzar associant els labels
	model -- model KMeans entrenat

	Returns: array de diccionaris amb l'assignació dels tipus als labels
	"""
	# temps_pujada, temps_baixada
	dicc = {'temps_pujada':0, 'temps_baixada': 1}

	logging.info('\nCentres:')
	for j in range(len(tipus)):
		logging.info("%s:\t(temps_pujada: %s\ttemps_baixada: %s)", j, model.cluster_centers_[j][dicc['temps_pujada']], model.cluster_centers_[j][dicc['temps_baixada']])


	# Procés d'assignació
	ind_label_0 = -1
	ind_label_1 = -1
	ind_label_2 = -1
	ind_label_3 = -1

	suma_max = 0
	suma_min = 50000

	for j, center in enumerate(clustering_model.cluster_centers_):
		suma = round(center[dicc['temps_pujada']], 1) + round(center[dicc['temps_baixada']], 1)
		if suma_max < suma:
			suma_max = suma
			ind_label_3 = j
		if suma_min > suma:
			suma_min = suma
			ind_label_0 = j

	tipus[0].update({'label': ind_label_0})
	tipus[3].update({'label': ind_label_3})

	lst = [0, 1, 2, 3]
	lst.remove(ind_label_0)
	lst.remove(ind_label_3)

	if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
		ind_label_1 = lst[0]
		ind_label_2 = lst[1]
	else:
		ind_label_1 = lst[1]
		ind_label_2 = lst[0]

	tipus[1].update({'label': ind_label_1})
	tipus[2].update({'label': ind_label_2})

	# Reordenació de labels per claritat
	"""
	Amb la reordenació en lloc d'obtenir:
	[{'name': 'BEBB', 'label': 0}, {'name': 'BEMB', 'label': 2}, {'name': 'MEBB', 'label': 3}, {'name': 'MEMB', 'label': 1}]
	
	Forçem a que sigui
	BEBB=0, BEMB=1, MEBB=2, MEMB=3
	"""
	# Guardem la correspondencia actual
	current_labels = [t['label'] for t in tipus]
	mapping = {current_labels[i]: i for i in range(4)}
	for t in tipus:
		t['label'] = mapping[t['label']]

	logging.info('\nHem fet l\'associació')
	logging.info('\nTipus i labels:\n%s', tipus)

	return tipus

def generar_informes(df, tipus):
	"""
	Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
	4 fitxers de ciclistes per cadascun dels clústers

	arguments:
		df -- dataframe
		tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

	Returns: None
	"""
	ciclistes_label = [
		df[df['label'] == 0],
		df[df['label'] == 1],
		df[df['label'] == 2],
		df[df['label'] == 3]
	]

	try:
		os.makedirs(os.path.dirname('informes/'))
	except FileExistsError:
		pass
	for tip in tipus:
		fitxer = tip['name'].replace(' ', '_') + '.txt'
		foutput = open("informes/" + fitxer, "w")
		t = [t for t in tipus if t['name'] == tip['name']]
		indexs = ciclistes_label[t[0]['label']].index
		for i in indexs:
			foutput.write(str(i) + '\n')
		foutput.close()

	logging.info('S\'han generat els informes en la carpeta informes/\n')

def nova_prediccio(dades, model):
	"""
	Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

	arguments:
		dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
		model -- clustering model
	Returns: (dades agrupades, prediccions del model)
	"""
	df_nous_ciclistes = pd.DataFrame(columns=['id', 'temps_pujada', 'temps_baixada', 'temps_total'], data=dades)
	df_nous_ciclistes_cleaned = clean(df_nous_ciclistes)

	logging.info('\nNous valors:\n%s', df_nous_ciclistes_cleaned[:3])
	return df_nous_ciclistes_cleaned, model.predict(df_nous_ciclistes_cleaned)

# ----------------------------------------------

if __name__ == "__main__":

	# Informació de logging
	logging.basicConfig(format='%(message)s', level=logging.DEBUG) # canviar a DEBUG mentre es programa
	logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR) # per tal de què el matplotlib no vomiti molts missatges

	path_dataset = './data/ciclistes.csv'
	ciclistes_data = load_dataset(path_dataset)

	EDA(ciclistes_data)

	ciclistes_data = clean(ciclistes_data)

	true_labels = extract_true_labels(ciclistes_data)

	ciclistes_data = ciclistes_data.drop('tipus', axis=1)
	visualitzar_pairplot(ciclistes_data)

	# Clustering amb KMeans
	selected_data = ciclistes_data[["temps_pujada", "temps_baixada"]]
	logging.debug('\nDades per l\'entrenament:\n%s\n...', selected_data[:3])

	clustering_model = clustering_kmeans(selected_data)
	# Guardem el model
	with open('model/clustering_model.pkl', 'wb') as f:
		pickle.dump(clustering_model, f)
	data_labels = clustering_model.labels_

	# Mostrar i guardar scores
	logging.info('\nHomogeneity: %.3f', homogeneity_score(true_labels, data_labels))
	logging.info('Completeness: %.3f', completeness_score(true_labels, data_labels))
	logging.info('V-Measure: %.3f', v_measure_score(true_labels, data_labels))
	with open('model/scores.pkl', 'wb') as f:
		pickle.dump({
			"h": homogeneity_score(true_labels, data_labels),
			"c": completeness_score(true_labels, data_labels),
			"v": v_measure_score(true_labels, data_labels)
		}, f)

	visualitzar_clusters(selected_data, data_labels)

	# array de diccionaris que assignarà els tipus als labels
	tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

	# Columna labels al dataframe
	ciclistes_data['label'] = clustering_model.labels_.tolist()
	logging.debug('\nColumna label:\n%s', ciclistes_data[:5])

	tipus = associar_clusters_patrons(tipus, clustering_model)

	# Guardem la variable tipus
	with open('model/tipus_dict.pkl', 'wb') as f:
		pickle.dump(tipus, f)
	logging.info('\nTipus i labels guardades: \n%s', tipus)

	# Generar informes
	generar_informes(ciclistes_data, tipus)
	
	# Classificació de nous valors
	nous_ciclistes = [
		[500, 3230, 1430, 4670], # BEBB
		[501, 3300, 2120, 5420], # BEMB
		[502, 4010, 1510, 5520], # MEBB
		[503, 4350, 2200, 6550] # MEMB
	]

	"""
	Durant les proves, el model no ha predit bé i unicament ha encertat el primer nou valor
	"""

	logging.debug('\nNous valors:\n%s', nous_ciclistes)
	df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, clustering_model)
	logging.info('\nPredicció dels valors:\n%s', pred)

	# Assignació dels nous valors als tipus
	for i, p in enumerate(pred):
		t = [t for t in tipus if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
