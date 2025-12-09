"""
@ IOC - CE IABD
"""
import unittest
import os
import sys
import pickle
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generardataset import generar_dataset
from clustersciclistes import (
    load_dataset, clean, extract_true_labels,
    clustering_kmeans, homogeneity_score,
    completeness_score, v_measure_score
)

class TestGenerarDataset(unittest.TestCase):
    """
    classe TestGenerarDataset
    """

    def setUp(self):
        self.mu_p_be = 3240
        self.mu_p_me = 4268
        self.mu_b_bb = 1440
        self.mu_b_mb = 2160
        self.sigma = 240

        self.dicc = [
            {"name": "BEBB", "mu_p": self.mu_p_be, "mu_b": self.mu_b_bb, "sigma": self.sigma},
            {"name": "BEMB", "mu_p": self.mu_p_be, "mu_b": self.mu_b_mb, "sigma": self.sigma},
            {"name": "MEBB", "mu_p": self.mu_p_me, "mu_b": self.mu_b_bb, "sigma": self.sigma},
            {"name": "MEMB", "mu_p": self.mu_p_me, "mu_b": self.mu_b_mb, "sigma": self.sigma}
        ]

        os.makedirs('tests/data', exist_ok=True)

    def cargar_csv(self):
        return pd.read_csv('tests/data/ciclistes.csv', sep=';')

    def get_col(self, df, posibles):
        """Devuelve la primera columna existente de una lista."""
        for col in posibles:
            if col in df.columns:
                return col
        raise KeyError(f"Ninguna de estas columnas existe: {posibles}")

    def test_longituddataset(self):
        generar_dataset(200, 1, self.dicc[0], 'tests/data/ciclistes.csv')
        df = self.cargar_csv()
        self.assertEqual(len(df), 200)

    def test_valorsmitjatp(self):
        generar_dataset(100, 1, self.dicc[0], 'tests/data/ciclistes.csv')
        df = self.cargar_csv()

        col_tp = self.get_col(df, ["temps_pujada", "tp"])
        tp_mig = df[col_tp].mean()

        self.assertLess(tp_mig, 3400)

    def test_valorsmitjatb(self):
        generar_dataset(100, 1, self.dicc[1], 'tests/data/ciclistes.csv')
        df = self.cargar_csv()

        col_tb = self.get_col(df, ["temps_baixada", "tb"])
        tb_mig = df[col_tb].mean()

        self.assertGreater(tb_mig, 2000)

class TestClustersCiclistes(unittest.TestCase):
    """
    classe TestClustersCiclistes
    """

    @classmethod
    def setUpClass(cls):
        path_dataset = 'tests/data/ciclistes.csv'

        # Asegurar dataset inicial para evitar errores
        if not os.path.exists(path_dataset):
            os.makedirs('tests/tests/data', exist_ok=True)
            generar_dataset(200, 1, {"name": "BEBB", "mu_p": 3240, "mu_b": 1440, "sigma": 240}, path_dataset)

        cls.ciclistes_data = load_dataset(path_dataset)
        cls.ciclistes_data_clean = clean(cls.ciclistes_data)
        cls.true_labels = extract_true_labels(cls.ciclistes_data_clean)
        cls.ciclistes_data_clean = cls.ciclistes_data_clean.drop('tipus', axis=1)

        cls.clustering_model = clustering_kmeans(cls.ciclistes_data_clean)
        cls.data_labels = cls.clustering_model.labels_

        os.makedirs('tests/model', exist_ok=True)
        with open('tests/model/clustering_model.pkl', 'wb') as f:
            pickle.dump(cls.clustering_model, f)

    def test_check_column(self):
        """
        Comprovem que una columna existeix
        """
        self.assertIn('temps_pujada', self.ciclistes_data_clean.columns)

    def test_data_labels(self):
        """
        Comprovem que data_labels té la mateixa longitud que ciclistes
        """
        self.assertEqual(len(self.data_labels), len(self.ciclistes_data_clean))

    def test_model_saved(self):
        """
        Comprovem que el fitxer clustering_model.pkl existeix
        """
        self.assertTrue(os.path.isfile('tests/model/clustering_model.pkl'))


if __name__ == '__main__':
    unittest.main()
