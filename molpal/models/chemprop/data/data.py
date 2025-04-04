import threading
from random import Random
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler
from ..features import BatchMolGraph, MolGraph

# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self, smiles: List[str], targets: List[Optional[float]] = None):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param row: The raw CSV row containing the information for this molecule.
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param features_generator: A list of features generators to use.
        """
        self.smiles = smiles
        self.targets = targets

        self.raw_targets = self.targets

    @property
    def mol(self) -> List[Chem.Mol]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = [SMILES_TO_MOL.get(s, Chem.MolFromSmiles(s)) for s in self.smiles]

        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m

        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_targets(self):
        """Resets the targets to their raw values."""
        self.targets = self.raw_targets


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()
        self._targets = np.array([d.targets for d in self._data], "f4")

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphss = [[MolGraph(m) for m in d.mol] for d in self._data]

            self._batch_graph = [
                BatchMolGraph([g[i] for g in mol_graphss]) for i in range(len(mol_graphss[0]))
            ]

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def targets(self) -> np.ndarray:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return self._targets

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return (
            len(self._data[0].features)
            if len(self._data) > 0 and self._data[0].features is not None
            else None
        )

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        :return: The size of the additional atom descriptor vector.
        """
        return (
            len(self._data[0].atom_descriptors[0])
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None
            else None
        )

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        :return: The size of the additional atom feature vector.
        """
        return (
            len(self._data[0].atom_features[0])
            if len(self._data) > 0 and self._data[0].atom_features is not None
            else None
        )

    def normalize_features(
        self, scaler: StandardScaler = None, replace_nan_token: int = 0
    ) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.

        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.

        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        if scaler is not None:
            self._scaler = scaler

        elif self._scaler is None:
            features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)

        for d in self._data:
            d.set_features(self._scaler.transform(d.raw_features.reshape(1, -1))[0])

        return self._scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def scale_targets(self, scaler: StandardScaler):
        targets = [d.raw_targets for d in self._data]
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self._data) == len(targets)

        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: an index or a slice of indices to get
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[idx]


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(
        self,
        dataset: MoleculeDataset,
        class_balance: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an
        equal number of positive and negative molecules). Set shuffle to True
        in order to get a random subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super().__init__(dataset)

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array(
                [any(target == 1 for target in datapoint.targets) for datapoint in dataset]
            )

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            idxs = [
                idx for pair in zip(self.positive_indices, self.negative_indices) for idx in pair
            ]
        else:
            idxs = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(idxs)

        return iter(idxs)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""Constructs a :class:`MoleculeDataset` from a list of
    :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph`
    for the constructed :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the
        :class:`MoleculeDatapoint`\ s.
    """
    dset = MoleculeDataset(data)
    dset.batch_graph()

    componentss = [bmg.get_components() for bmg in dset.batch_graph()]
    return componentss, torch.from_numpy(dset.targets())


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` that loads
    :class:`MoleculeDatapoint`s from a :class:`MoleculeDataset`."""

    def __init__(
        self,
        dataset: MoleculeDataset,
        batch_size: int = 50,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler = None,
        batch_sampler = None,
        **kwargs
    ):
        """
        :param dataset: The :class:`MoleculeDataset` containing the :class:`MoleculeDatapoint`s.
        :param batch_size: The batch size.
        :param num_workers: The number of workers to use to load the data.
        :param pin_memory: Whether to pin the memory.
        :param shuffle: Whether to shuffle the data.
        :param drop_last: Whether to drop the last incomplete batch.
        :param sampler: Optional sampler for the data.
        :param batch_sampler: Optional batch sampler for the data.
        :param kwargs: Additional arguments to pass to the DataLoader.
        """
        self.dataset = dataset
        
        # filter out any collate_fn that might be in kwargs to avoid conflicts
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'collate_fn'}

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=construct_molecule_batch,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            batch_sampler=batch_sampler,
            **kwargs_filtered
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration
        through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    # def __iter__(self) -> Iterator[MoleculeDataset]:
    #     r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
    #     return super(MoleculeDataLoader, self).__iter__()
