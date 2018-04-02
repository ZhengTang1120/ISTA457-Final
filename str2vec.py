from typing import Sequence, Any

import numpy as np


class Index:

    def __init__(self, vocab: Sequence[Any], start=1):
        """
        Assigns an index to each unique word in the `vocab` iterable,
        with indexes starting from `start`.
        """
        self.obj_to_idx = dict()
        self.idx_to_obj = dict()
        self.start = start
        for obj in vocab:
            if obj not in self.obj_to_idx:
                self.idx_to_obj[len(self.obj_to_idx)+start] = obj
                self.obj_to_idx[obj] = len(self.obj_to_idx)+start

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, use `start-1` as the index.

        :param object_seq: A sequence of objects
        :return: A 1-dimensional array of the object indexes.
        """
        res = np.zeros(len(object_seq))
        for i, obj in enumerate(object_seq):
            if obj in self.obj_to_idx:
                res[i] = self.obj_to_idx[obj]
            else:
                res[i] = self.start-1
        return res

    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        :param object_seq_seq: A sequence of sequences of objects
        :return: A 2-dimensional array of the object indexes.
        """
        row_length = max([len(seq) for seq in object_seq_seq])
        res = np.zeros((len(object_seq_seq), row_length))
        for i, object_seq in enumerate(object_seq_seq):
            for j, obj in enumerate(object_seq):
                if obj in self.obj_to_idx:
                    res[i][j] = self.obj_to_idx[obj]
                else:
                    res[i][j] = self.start-1
        return res



    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes
        """
        res = np.zeros(len(self.idx_to_obj)+1)
        for obj in object_seq:
            if obj in self.obj_to_idx:
                res[self.obj_to_idx[obj]] = 1
        return res

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes
        """
        res = np.zeros((len(object_seq_seq), len(self.idx_to_obj)))
        for i, object_seq in enumerate(object_seq_seq):
            for obj in object_seq:
                if obj in self.obj_to_idx:
                    res[i][self.obj_to_idx[obj]] = 1
        return res

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        return [self.idx_to_obj[idx] for idx in index_vector if idx in self.idx_to_obj]

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        return [[self.idx_to_obj[idx] for idx in idxs  if idx in self.idx_to_obj] for idxs in index_matrix]

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        return [self.idx_to_obj[i] for i, value in enumerate(vector) if value == 1]

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        return [[self.idx_to_obj[i] for i, value in enumerate(vector) if value == 1] for vector in binary_matrix]
