import os
import logging
import numpy as np
import random
import csv

# Creació del logger
logger = logging.getLogger("generardataset")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG) # canviar a DEBUG mentre es programa
logger.addHandler(ch)

def generar_dataset(num, ind, dicc, str_ciclistes='data/ciclistes.csv'):
	"""
	Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
	num és el número de files/ciclistes a generar. ind és l'index/identificador/dorsal.
	"""

	# Obrim el fitxer str_ciclistes
	with open(str_ciclistes, "w", newline='') as f:
		logger.info('Obrint/Creant el fitxer de ciclistes...')
		writer = csv.writer(f, delimiter=';') # si no, utilitza com separador ','
		writer.writerow(["id", "tipus", "temps_pujada", "temps_baixada", "temps_total"])

		for i in range(num):
			tipus = random.choice(dicc)

			# Generació de temps
			temps_p = max(0, int(np.random.normal(tipus["mu_p"], tipus["sigma"])))
			temps_b = max(0, int(np.random.normal(tipus["mu_b"], tipus["sigma"])))
			total = temps_p + temps_b
			
			writer.writerow([ind + i, tipus["name"], temps_p, temps_b, total])
		return str_ciclistes

if __name__ == "__main__":

	str_ciclistes = 'data/ciclistes.csv'

	random.seed(42) # Per reproduir resultats
	np.random.seed(42)

	try:
		os.makedirs(os.path.dirname(str_ciclistes))
	except FileExistsError:
		pass

	# BEBB: bons escaladors, bons baixadors
	# BEMB: bons escaladors, mal baixadors
	# MEBB: mal escaladors, bons baixadors
	# MEMB: mal escaladors, mal baixadors

	# Port del Cantó (18 Km de pujada, 18 Km de baixada)
	# pujar a 20 Km/h són 54 min = 3240 seg
	# pujar a 14 Km/h són 77 min = 4268 seg
	# baixar a 45 Km/h són 24 min = 1440 seg
	# baixar a 30 Km/h són 36 min = 2160 seg
	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]

	# Generem el dataset
	generar_dataset(1000, 1, dicc, str_ciclistes)

	logger.info("S'ha generat data/ciclistes.csv")
