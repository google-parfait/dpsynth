.. Copyright 2026 Google LLC
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

#############
API Reference
#############

This page documents the public Python API for DPSynth. The library is organized
into three layers:

- **Domain specification** — describing the schema of your tabular data.
- **Constraints** — optional cross-attribute restrictions on generated values.
- **Mechanisms** — configuring and running differentially private synthesis.

.. contents:: On this page
   :local:
   :depth: 2

----

Domain Specification (``dpsynth.domain``)
==========================================

.. currentmodule:: dpsynth.domain

The ``domain`` module provides dataclasses for describing the schema of a
tabular dataset.  Each column is represented by one of the attribute types
below.  Pass a mapping of column names to attribute objects as the ``domains``
argument to :class:`~dpsynth.TabularConfig`.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   CategoricalAttribute
   NumericalAttribute
   OpenSetCategoricalAttribute
   FreeFormTextAttribute

----

Cross-Attribute Constraints (``dpsynth.constraints``)
======================================================

.. currentmodule:: dpsynth.constraints

The ``constraints`` module lets you express known relationships between columns
so that the synthetic data honours them.  Pass a list of
:class:`~dpsynth.constraints.Constraint` objects as
``cross_attribute_constraints`` to :class:`~dpsynth.TabularConfig`.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   Constraint

----

Mechanism Abstractions (``dpsynth.api``)
=========================================

.. currentmodule:: dpsynth.api

These abstract base classes define the three-phase *construct → calibrate → run*
protocol shared by all DPSynth mechanisms.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   MechanismConfig
   CalibratedMechanism

----

Privacy Calibration & Reporting (``dpsynth``)
=============================================

.. currentmodule:: dpsynth

Top-level utilities for calibrating a :class:`~dpsynth.api.MechanismConfig` to
a target ``(epsilon, delta)``-DP guarantee and summarizing the resulting
privacy profile across evaluation points.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   calibrate

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   PrivacyReport

----

Tabular Synthesis (``dpsynth``)
===============================

.. currentmodule:: dpsynth

The primary entry point for generating differentially private synthetic data from standard tabular datasets (such as Pandas DataFrames).

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   TabularConfig
   TabularMechanism

----

Discrete Mechanisms (``dpsynth.discrete_mechanisms``)
======================================================

.. currentmodule:: dpsynth.discrete_mechanisms

Discrete mechanisms operate on pre-discretized integer datasets
(:class:`mbi.Dataset`).  :class:`~dpsynth.TabularConfig` applies them internally
after encoding your DataFrame.  Use them directly only if you already have a
discrete dataset.

Mechanism Configs
-----------------

Each config class corresponds to a published DP synthesis algorithm.  Pass one
as the ``discrete_mechanism`` argument to :class:`~dpsynth.TabularConfig`,
or use :class:`~dpsynth.discrete_mechanisms.DiscreteConfig` to add one-way
marginal measurement and domain compression.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   AIMConfig
   MSTConfig
   IndependentConfig
   DirectConfig
   SWIFTConfig

DiscreteConfig and DiscreteMechanism
------------------------------------

:class:`DiscreteConfig` wraps any of the mechanism configs above with one-way
marginal pre-measurement and optional domain compression. When calibrated, it
produces a runnable :class:`DiscreteMechanism`. This is the recommended entry
point when you have a pre-discretized table.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   DiscreteConfig
   DiscreteMechanism
