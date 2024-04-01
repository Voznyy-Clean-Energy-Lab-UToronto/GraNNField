"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

import torch
from ase.calculators.calculator import Calculator, all_changes

from grannfield.data.materials import AtomsConverter
from grannfield.utils.units import Units


class GraNNFieldCalculatorError(Exception):
    pass


class GraNNFieldCalculator(Calculator):
    """
    ASE md for schnetpack machine learning models.

    Args:
        ml_model (schnetpack.AtomisticModel): Trained modules for
            calculations
        device (str): select to run calculations on 'cuda' or 'cpu'
        collect_triples (bool): Set to True if angular features are needed,
            for example, while using 'wascf' models
        environment_provider (callable): Provides neighbor lists
        pair_provider (callable): Provides list of neighbor pairs. Only
            required if angular descriptors are used. Default is none.
        **kwargs: Additional arguments for basic ase md class
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
            self,
            base_model,
            forces_model=None,
            stress_model=None,
            device='cpu',
            normalizers=None,
            energy=None,
            forces=None,
            stress=None,
            apply_constraint=False,
            energy_units='eV',
            forces_units='eV/Angstrom',
            stress_units='eV/Angstrom/Angstrom/Angstrom',
            **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.base_model = base_model
        self.base_model.to(device)
        self.forces_model = forces_model
        self.stress_model = stress_model

        if self.forces_model is not None:
            self.forces_model.to(device)

        if self.stress_model is not None:
            self.stress_model.to(device)

        self.normalizers = normalizers
        self.apply_constraint = apply_constraint

        self.atoms_converter = AtomsConverter(
            device=device,
        )

        self.model_energy = energy
        self.model_forces = forces
        self.model_stress = stress

        # Convert to ASE internal units (energy=eV, length=A)
        self.energy_units = Units.unit2unit(energy_units, 'eV')
        self.forces_units = Units.unit2unit(forces_units, 'eV/Angstrom')
        self.stress_units = Units.unit2unit(stress_units, 'eV/A/A/A')

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original md to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if properties is None:
            properties = [self.model_energy]
        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)
            # Convert to schnetpack input format
            model_inputs = self.atoms_converter(atoms)
            # Call modules
            model_results = self.base_model(model_inputs)

            if self.forces_model is not None:
                forces_model_results = self.forces_model(model_inputs)
            elif self.model_forces is not None:
                forces_model_results = model_results

            if self.stress_model is not None:
                stress_model_results = self.stress_model(model_inputs)
            elif self.model_stress is not None:
                stress_model_results = model_results

            results = {}
            # Convert outputs to md format
            if self.model_energy is not None:
                if self.model_energy not in model_results.keys():
                    raise GraNNFieldCalculatorError(
                        "'{}' is not a property of your modules. Please "
                        "check the modules "
                        "properties!".format(self.model_energy)
                    )
                energy = model_results[self.model_energy]
                if self.model_energy in self.normalizers.keys():
                    energy = self.normalizers[self.model_energy].denorm(energy)
                energy = energy.cpu().data.numpy()
                results[self.model_energy] = (
                    energy.item() * self.energy_units
                )  # ase md should return scalar energy

            if self.model_forces is not None:
                if self.model_forces not in forces_model_results.keys():
                    raise GraNNFieldCalculatorError(
                        "'{}' is not a property of your modules. Please "
                        "check the modules"
                        "properties!".format(self.model_forces)
                    )
                forces = forces_model_results[self.model_forces]
                if self.model_forces in self.normalizers.keys():
                    forces = self.normalizers[self.model_forces].denorm(forces)
                forces = forces.cpu().data.numpy()
                if self.apply_constraint:
                    fixed_idx = torch.where(model_inputs['fixed'] == 1)[0].cpu().data.numpy()
                    forces[fixed_idx] = 0
                results[self.model_forces] = (
                    forces.reshape((len(atoms), 3)) * self.forces_units
                )

            if self.model_stress is not None:
                if atoms.cell.volume <= 0.0:
                    raise GraNNFieldCalculatorError(
                        'Cell with 0 volume encountered for stress computation'
                    )
                if self.model_stress not in stress_model_results.keys():
                    raise GraNNFieldCalculatorError(
                        '{} is not a property of your modules. Please '
                        'check the modules'
                        'properties! If desired, stress tensor computation can be '
                        'activated via schnetpack.utils.activate_stress_computation '
                        'at ones own risk.'.format(self.model_stress)
                    )
                stress = stress_model_results[self.model_stress].cpu().data.numpy()
                results[self.model_stress] = stress.reshape((3, 3)) * self.stress_units

            self.results = results