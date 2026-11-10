# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for post-processing noisy marginals into synthetic data.

These functions are useful in situations where the noisy measurements are
obtained from an external system, and you want to leverage the capabilities
of this library to post-process them to resolve inconsitencies and generate
synthtetic data.
"""

import collections
from collections.abc import Sequence
from typing import Any

from dpsynth import transformations
from dpsynth.core import constraints
from dpsynth.core import domain
import mbi
import numpy as np
import pandas as pd


def _check_format(marginals: Sequence[pd.DataFrame]):
  for df in marginals:
    if df.columns[-1] != 'count':
      raise ValueError(
          'The last column of the marginals must be "count", found'
          f' {df.columns[-1]} instead.'
      )


def infer_categorical_domain(
    marginals: Sequence[pd.DataFrame],
    extra_domain_elements: dict[str, Sequence[Any]] | None = None,
) -> dict[str, domain.CategoricalAttribute]:
  """Infer the domain from marginals by inspecting the attribute values present.

  Note: When passing in noisy marginals computed from a DP mechanism, this
  function is a pure post-processing step, and does not require any privacy
  budget.

  Args:
    marginals: A Sequence of marginals, each of which is a dataframe, where the
      first k columns are the attribute values and the last column is the count.
    extra_domain_elements: Extra domain elements that should be considered
      in-domain, even if they were not directly measured.

  Returns:
    A dictionary mapping attribute names to CategoricalAttributes.
  """
  _check_format(marginals)
  if extra_domain_elements is None:
    extra_domain_elements = {}
  possible_values = collections.defaultdict(
      set, {k: set(v) for k, v in extra_domain_elements.items()}
  )
  for df in marginals:
    for col in df.columns[:-1]:
      possible_values[col] |= set(df[col].unique())

  return {
      col: domain.CategoricalAttribute(
          possible_values=list(possible_values[col]),  # pyrefly: ignore[unexpected-keyword]
      )
      for col in possible_values.keys()
  }


def encode_marginals(
    marginals: Sequence[pd.DataFrame],
    attribute_domains: dict[str, domain.CategoricalAttribute],
) -> Sequence[mbi.Factor]:
  """Convert the marginals into mbi.Factor objects.

  This function converts the categorical values into integer codes, and
  then converts the pandas dataframe into a multi-dimensoinal array, filling
  in missing values with zeros as necessary.

  Args:
    marginals: A list of marginals, each of which is a dataframe, where the
      first k columns are the attribute values and the last column is the count.
    attribute_domains: A dictionary mapping attribute names to
      CategoricalAttributes.

  Returns:
    A list of mbi.Factor objects, one for each marginal.
  """
  _check_format(marginals)
  transform_fns = {
      attribute_name: transformations.discrete_encoder(
          attribute_domains[attribute_name]
      )
      for attribute_name in attribute_domains
  }

  mbi_domain = mbi.Domain.fromdict({
      attribute_name: attribute_domains[attribute_name].size
      for attribute_name in attribute_domains
  })

  encoded = []
  for df in marginals:
    attributes = list(df.columns[:-1])
    df2 = pd.DataFrame()
    for col in attributes:
      df2[col] = df[col].map(transform_fns[col]).astype(int)
    df2['count'] = df['count'].astype(float).fillna(0.0)

    if not attributes:
      marginal = np.array(df2['count'].iloc[0])
    else:
      marginal = np.full(mbi_domain.project(attributes).shape, np.nan)
      indices, counts = df2[attributes].values.T, df2['count'].values
      marginal[*indices] = counts

    marginal = mbi.Factor(
        domain=mbi_domain.project(attributes), values=marginal  # pyrefly: ignore[bad-argument-type]
    )

    encoded.append(marginal)

  return encoded


def generate_synthetic_data_from_marginals(
    noisy_marginals: Sequence[pd.DataFrame],
    iters: int = 10000,
    log: bool = False,
    nrows: int | None = None,
    estimator: mbi.Estimator | None = None,
    marginal_oracle: mbi.marginal_oracles.MarginalOracle | None = None,
    exact_marginals: Sequence[pd.DataFrame] | None = None,
    cross_attribute_constraints: Sequence[constraints.Constraint] = (),
    extra_domain_elements: dict[str, Sequence[Any]] | None = None,
) -> pd.DataFrame:
  """Estimate a graphical model from the noisy marginals.

  This function uses Private-PGM to solve the L2 minimization problem:

    min_{p} || Q(p) - y ||_2^2

  where y is the concatenated vector of noisy marginals, Q is the function
  that maps a probability distribution to its marginals, and p is optimized
  over the set of probability distributions within the class of undirected
  graphical models (markov random fields).

  Note on domain discovery: The categorical domain is automatically inferred
  from the noisy marginals by taking the union of obesrved attribute values
  across all noisy marginals.  If a specific attribute value is observed in one
  noisy marginal but not another, we assume it was unmeasured, rather than
  assuming it was measured with a zero count. This matches the behavior
  of federated analytics SQL pipelines, for instance.

  Args:
    noisy_marginals: A list of marginals, each of which is a dataframe, where
      the first k columns are the attribute values and the last column is the
      count.
    iters: The number of iterations to run the optimization for.
    log: Whether to log the progress of the optimization.
    nrows: The number of rows of synthetic data to generate. If None, the number
      of rows will be estimated from the marginals.
    estimator: The estimator to use for the optimization.
    marginal_oracle: The marginal oracle to use for the optimization.
    exact_marginals: A list of exact marginals in the same format as noisy
      marginals. If provided, the error between the estimated marginals and the
      exact marginals will be logged during optimization.
    cross_attribute_constraints: Constraints to enforce on the generated data.
    extra_domain_elements: Extra domain elements that should be considered
      in-domain, even if they were not directly measured.

  Returns:
    A DataFrame containing synthetic data that approximately minimizes the
    L2 distance between the noisy marginals and the synthetic data marginals.
  """
  _check_format(noisy_marginals)
  attribute_domains = infer_categorical_domain(
      exact_marginals or noisy_marginals, extra_domain_elements
  )

  mbi_domain = mbi.Domain.fromdict(
      {col: attribute_domains[col].size for col in attribute_domains}
  )
  noisy_encoded = encode_marginals(noisy_marginals, attribute_domains)

  measurements = []
  estimated_total = 1
  for noisy in noisy_encoded:

    # Unmeasured attribute combinations are set to zero (via zero weights),
    # ensuring they do not contribute to the loss.
    mask = np.isnan(noisy.values.flatten())
    weights = np.where(mask, 0.0, 1.0)
    answer = np.where(mask, 0.0, noisy.values.flatten())

    # Because we do not necessarily measure the entire marginal, mbi cannot
    # reliably estimate the number of records, so we use this heuristic instead.
    estimated_total = max(estimated_total, float(answer.sum()))

    measurements.append(
        mbi.LinearMeasurement(
            noisy_measurement=answer,
            clique=noisy.domain.attributes,
            query=mbi.WeightedQuery(weights),
        )
    )

  duck_typed_exact_data = None
  if exact_marginals is not None:
    exact_encoded = encode_marginals(exact_marginals, attribute_domains)
    factors = {}
    for x in exact_encoded:
      factors[x.domain.attributes] = x
    duck_typed_exact_data = mbi.CliqueVector(
        mbi_domain,
        list(factors.keys()),
        factors,
    )

  mbi_constraints = tuple(c.to_mbi() for c in cross_attribute_constraints)

  if log:
    callback_fn = mbi.callbacks.default(
        measurements, mbi_domain, data=duck_typed_exact_data  # pyrefly: ignore[bad-argument-type]
    )
  else:
    callback_fn = lambda _: None
  if estimator is None:
    estimator = mbi.estimation.MirrorDescent(marginal_oracle=marginal_oracle)
  model = estimator.estimate(
      mbi_domain,
      measurements,
      iters=iters,
      callback_fn=callback_fn,
      known_total=estimated_total,
      constraints=mbi_constraints,
  )
  encoded_data = model.synthetic_data(nrows)

  result = pd.DataFrame()
  for col in attribute_domains:
    attr_domain = attribute_domains[col]
    transform_fn = transformations.discrete_encoder(attr_domain)
    result[col] = [transform_fn.inverse(x) for x in encoded_data.to_dict()[col]]
  return result
