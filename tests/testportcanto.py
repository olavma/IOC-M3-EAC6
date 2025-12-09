"""
@ IOC - CE IABD
"""
import unittest
import os
import sys
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

class TestGenerarDataset(unittest.TestCase):
	"""
	classe TestGenerarDataset
	"""
    def setUp(self):
        self.mu_p_be = 3240 # mitjana temps pujada bons escaladors
        self.mu_p_me = 4268 # mitjana temps pujada mals escaladors
        self.mu_b_bb = 1440 # mitjana temps baixada bons baixadors
        self.mu_b_mb = 2160 # mitjana temps baixada mals baixadors
        self.sigma = 240 # 240 s = 4 min
        self.dicc = [
            {"name":"BEBB", "mu_p": self.mu_p_be, "mu_b": self.mu_b_bb, "sigma": self.sigma},
            {"name":"BEMB", "mu_p": self.mu_p_be, "mu_b": self.mu_b_mb, "sigma": self.sigma},
            {"name":"MEBB", "mu_p": self.mu_p_me, "mu_b": self.mu_b_bb, "sigma": self.sigma},
            {"name":"MEMB", "mu_p": self.mu_p_me, "mu_b": self.mu_b_mb, "sigma": self.sigma}
        ]
        # Asegurarse de que la carpeta data existe
        os.makedirs('tests/data', exist_ok=True)

    def test_longituddataset(self):
		"""
		Test la longitud de l'array
		"""
        arr = generar_dataset(200, 1, self.dicc[0], 'tests/data/ciclistes.csv')
        self.assertEqual(len(arr), 200)

    def test_valorsmitjatp(self):
		"""
		Test del valor mitjà del tp
		"""
        arr = generar_dataset(100, 1, self.dicc[0], 'tests/data/ciclistes.csv')
		arr_tp = [row[2] for row in arr]  # columna temps_pujada
		tp_mig = sum(arr_tp)/len(arr_tp)
        self.assertLess(tp_mig, 3400)

    def test_valorsmitjatb(self):
		"""
		Test del valor mitjà del tp
		"""
        arr = generar_dataset(100, 1, self.dicc[1], 'tests/data/ciclistes.csv')
        arr_tb = [row[3] for row in arr]  # columna temps_baixada
        tb_mig = sum(arr_tb)/len(arr_tb)
        self.assertGreater(tb_mig, 2000)

class TestClustersCiclistes(unittest.TestCase):
	"""
	classe TestClustersCiclistes
	"""
    @classmethod
    def setUpClass(cls):
        # Preparar datos y modelo
        path_dataset = 'tests/data/ciclistes.csv'
        cls.ciclistes_data = load_dataset(path_dataset)
        cls.ciclistes_data_clean = clean(cls.ciclistes_data)
        true_labels = extract_true_labels(cls.ciclistes_data_clean)
        cls.ciclistes_data_clean = cls.ciclistes_data_clean.drop('tipus', axis=1)
        cls.clustering_model = clustering_kmeans(cls.ciclistes_data_clean)
        os.makedirs('model', exist_ok=True)
        with open('model/clustering_model.pkl', 'wb') as f:
            pickle.dump(cls.clustering_model, f)
        cls.data_labels = cls.clustering_model.labels_

    def test_check_column(self):
		"""
		Comprovem que una columna existeix
		"""
        self.assertIn('tp', self.ciclistes_data_clean.columns)

    def test_data_labels(self):
		"""
		Comprovem que data_labels té la mateixa longitud que ciclistes
		"""

        self.assertEqual(len(self.data_labels), len(self.ciclistes_data_clean))

    def test_model_saved(self):
		"""
		Comprovem que a la carpeta model/ hi ha els fitxer clustering_model.pkl
		"""
        check_file = os.path.isfile('model/clustering_model.pkl')
        self.assertTrue(check_file)

# -----------------------------
if __name__ == '__main__':
    unittest.main()
