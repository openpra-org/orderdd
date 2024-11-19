import json
from typing import Tuple, Any
import sqlite3
import re
import dd.autoref as autoref
from dd.autoref import BDD, Function


class BDDDB:

    def __init__(self, db_path='bdd_sizes.db', force_build=False):
        super().__init__()
        self._db_path = db_path
        self._connection = self.init_db()
        self._force_build = force_build

    def __del__(self):
        if self._connection:
            self._connection.close()

    def init_db(self):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bdd_data (
                expression TEXT,
                var_order TEXT,
                size INTEGER,
                UNIQUE(expression, var_order)
            )
        ''')
        conn.commit()
        return conn

    def lookup(self, expression, var_order):
        cursor = self._connection.cursor()
        cursor.execute('''
            SELECT size FROM bdd_data WHERE expression = ? AND var_order = ?
        ''', BDDDB.sanitize(expression, var_order))
        result = cursor.fetchone()
        return result[0] if result else None

    def insert(self, expression, var_order, size):
        cursor = self._connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO bdd_data (expression, var_order, size) VALUES (?, ?, ?)
        ''', (*BDDDB.sanitize(expression, var_order), size))
        self._connection.commit()

    @staticmethod
    def build(expr: str, var_order: list[str]) -> tuple[BDD, Any]:
        bdd = autoref.BDD()
        for var in var_order:
            bdd.add_var(var)
        bdd_expr = bdd.add_expr(expr)
        return bdd, bdd_expr

    def get_size(self, expr: str, var_order: list[str], reorder: bool = False) -> int:
        # if re-ordering, we have to build the BDD anyway.
        if reorder:
            bdd, bdd_expr = BDDDB.build(expr, var_order)
            bdd.reorder()
            reordered_vars = sorted(bdd.vars, key=bdd.vars.get)
            reordered_size = bdd_expr.dag_size
            #print("inserting re-ordered size", expr, reordered_vars, reordered_size, bdd)
            self.insert(expr, reordered_vars, reordered_size)
            return reordered_size

        if not self._force_build:
            cached_size = self.lookup(expr, var_order)
            if cached_size is not None:
                return cached_size

        # size is None
        bdd, bdd_expr = BDDDB.build(expr, var_order)
        computed_size = bdd_expr.dag_size
        #print("inserting size", expr, var_order, computed_size, bdd)
        self.insert(expr, var_order, computed_size)
        return computed_size

    @staticmethod
    def sanitize_list(_list: list[str]):
        # This regex will remove all double quotes, square brackets, and spaces
        return re.sub(r'[\[\]"]|\s+', '', json.dumps(_list), flags=re.UNICODE)

    @staticmethod
    def sanitize_str(_str: str):
        return _str.replace(" ", "")

    @staticmethod
    def sanitize(expression, var_order) -> Tuple[str, str]:
        return BDDDB.sanitize_str(expression), BDDDB.sanitize_list(var_order)
