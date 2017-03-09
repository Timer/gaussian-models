#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "bnet.hpp"
#include "matrix.hpp"
#include "rand.hpp"

#define VERBOSE 1
#define SAVE_NETWORKS 0

double dirichlet_score_family(SMatrix counts, SCPD cpd) {
  SMatrix ns = cpd->sizes, prior = cpd->dirichlet;
  SMatrix ns_self = ns->extract_indices(ns->rows - 1, ns->rows, 0, ns->cols);
  SMatrix pnc = counts + prior;
  SMatrix gamma_pnc = pnc->lgammaed(), gamma_prior = prior->lgammaed();
  SMatrix lu_mat = gamma_pnc - gamma_prior;
  SMatrix LU = lu_mat->sum_n_cols(ns_self->data[0]);
  SMatrix alpha_ij = prior->sum_n_cols(ns_self->data[0]);
  SMatrix N_ij = counts->sum_n_cols(ns_self->data[0]);
  SMatrix gamma_alpha = alpha_ij->lgammaed();
  SMatrix alpha_N = N_ij + alpha_ij;
  SMatrix gamma_alpha_N = alpha_N->lgammaed();
  SMatrix LV = gamma_alpha - gamma_alpha_N;
  SMatrix LU_LV = LU + LV;
  double score = LU_LV->sumAllValue();
  return score;
}

int count_index(SMatrix sz, SMatrix sample_data, int col) {
  SMatrix mat_col = sample_data->extract_indices(0, sample_data->rows, col, col + 1);
  int index = 0;
  for (int i = 0, m = 1; i < mat_col->rows * mat_col->cols; m *= sz->data[i++]) {
    index += ((mat_col->data[i]) - 1) * m;
  }
  return index;
}

SMatrix compute_counts(SMatrix data, SMatrix sz) {
  SMatrix count = std::make_shared<Matrix>(sz->multiplyAllValues(), 1);
  for (int i = 0; i < data->cols; ++i) {
    count->data[count_index(sz, data, i)] += 1;
  }
  return count;
}

double log_marg_prob_node(SCPD cpd, SMatrix self_ev, SMatrix pev) {
  SMatrix data = pev->concat_rows(self_ev, false);
  SMatrix counts = compute_counts(data, cpd->sizes);
  return dirichlet_score_family(counts, cpd);
}

SMatrix prob_node(SCPD cpd, SMatrix self_ev, SMatrix pev) {
  SMatrix sample_data = pev->concat_rows(self_ev, false);
  SMatrix prob = std::make_shared<Matrix>(sample_data->rows, sample_data->cols);
  for (int i = 0; i < sample_data->cols; ++i) {
    SMatrix mat_col = sample_data->extract_indices(0, sample_data->rows, i, i + 1);
    int index = 0;
    auto dd = cpd->sizes->data;
    for (int j = 0, m = 1; j < mat_col->rows * mat_col->cols; m *= dd[j++]) {
      index += ((mat_col->data[j]) - 1) * m;
    }
    prob->data[i] = cpd->cpt->data[index];
  }
  return prob;
}

double log_prob_node(SCPD cpd, SMatrix self_ev, SMatrix pev) {
  double score = 0;
  SMatrix p = prob_node(cpd, self_ev, pev);
  for (int i = 0; i < p->rows * p->cols; ++i) {
    double d = p->data[i];
    score += d <= 0 ? DBL_MIN : log(d);
  }
  return score;
}

SCPD tabular_CPD(SMatrix dag, SMatrix ns, int self) {
  SCPD cpd = std::make_shared<CPD>();
  std::vector<int> ps = dag->adjacency_matrix_parents(self);
  ps.push_back(self);
  SMatrix fam_sz = std::make_shared<Matrix>(ps.size(), 1);
  for (int i = 0; i < ps.size(); ++i) {
    fam_sz->data[i] = ns->data[ps[i]];
  }
  cpd->sizes = fam_sz;
  SMatrix calc = fam_sz->extract_indices(0, ps.size() - 1, 0, 1);
  int psz = calc->multiplyAllValues();
  cpd->dirichlet = std::make_shared<Matrix>(
      fam_sz->multiplyAllValues(), 1,
      (1.0 / psz) * (1.0 / ns->data[self]));
  cpd->cpt = nullptr;
  return cpd;
}

double score_family(int j, std::vector<int> ps, SMatrix ns, std::vector<int> discrete, SMatrix data,
                    std::string scoring_fn) {
  SMatrix dag = std::make_shared<Matrix>(data->rows, data->rows);
  if (ps.size() > 0) {
    dag->set_list_index(1, ps, j, j + 1);
    // TODO: sort `ps` here.
  }
  SMatrix data_sub_1 = data->extract_indices(j, j + 1, 0, data->cols),
          data_sub_2 = data->extract_list_index(ps, 0, data->cols);
  SCPD cpd = tabular_CPD(dag, ns, j);
  double score;
  if (scoring_fn == "bayesian") {
    score = log_marg_prob_node(cpd, data_sub_1, data_sub_2);
  } else if (scoring_fn == "bic") {
    std::vector<int> fam(ps);
    fam.push_back(j);
    SMatrix data_sub_3 = data->extract_list_index(fam, 0, data->cols);
    SMatrix counts = compute_counts(data_sub_3, cpd->sizes);
    cpd->cpt = counts + cpd->dirichlet;
    cpd->cpt->mk_stochastic(ns);
    double L = log_prob_node(cpd, data_sub_1, data_sub_2);
    SMatrix sz = cpd->sizes;
    const int len = sz->rows * sz->cols;
    const int value = sz->data[len - 1];
    sz->set_position(len, value - 1);
    score = L - 0.5 * sz->multiplyAllValues() * log(data->cols);
    sz->set_position(len, value);
  } else {
    throw "dead in the water, mate";
  }
  return score;
}

SMatrix learn_struct_K2(SMatrix data, SMatrix ns, std::vector<int> order, std::string scoring_fn, int max_parents) {
  assert(order.size() == data->rows);
  const int n = data->rows;
  int max_fan_in = max_parents == 0 ? n : max_parents;
  std::vector<int> discrete;
  for (int i = 0; i < n; ++i) discrete.push_back(i);

  SMatrix dag = std::make_shared<Matrix>(n, n);
  int parent_order = 0;
  for (int i = 0; i < n; ++i) {
    std::vector<int> ps;
    const int j = order[i];
    double score = score_family(j, ps, ns, discrete, data, scoring_fn);
#if VERBOSE
    printf("\nnode %d, empty score %6.4f\n", j, score);
#endif
    for (; ps.size() <= max_fan_in;) {
      std::vector<int> order_sub(order.begin(), order.begin() + i);
      std::vector<int> pps(order_sub.size());
      auto it = std::set_difference(order_sub.begin(), order_sub.end(), ps.begin(), ps.end(), pps.begin());
      pps.resize(it - pps.begin());
      int nps = pps.size();
      SMatrix pscore = std::make_shared<Matrix>(1, nps);
      for (int pi = 0; pi < nps; ++pi) {
        int p = pps[pi];
        ps.push_back(p);
        int n_index = ps.size() - 1;
        pscore->data[pi] = score_family(j, ps, ns, discrete, data, scoring_fn);
#if VERBOSE
        printf("considering adding %d to %d, score %6.4f\n", p, j, pscore->data[pi]);
#endif
        ps.erase(ps.begin() + n_index);
      }
      double best_pscore = -DBL_MAX;
      int best_p = -1;
      for (int i = 0; i < nps; ++i) {
        double d = pscore->data[i];
        if (d > best_pscore) {
          best_pscore = d;
          best_p = i;
        }
      }
      if (best_p == -1) {
        break;
      }
      best_p = pps[best_p];
      if (best_pscore > score) {
        score = best_pscore;
        ps.push_back(best_p);
#if VERBOSE
        printf("* adding %d to %d, score %6.4f\n", best_p, j, best_pscore);
#endif
      } else {
        break;
      }
    }
    if (ps.size() > 0) {
      dag->set_list_index(++parent_order, ps, j, j + 1);
    }
  }
  return dag;
}

int exec(int forkIndex, int forkSize, bool data_transposed, std::string f_data,
         int topologies, std::string f_output, std::string scoring_fn, int max_parents) {
  SMatrix data = load(f_data, !data_transposed),
          sz = data->create_sz();
  SMatrix orders = std::make_shared<Matrix>(data->rows * topologies, data->rows);
#if SAVE_NETWORKS
  SMatrix networks =
      std::make_shared<Matrix>(data->rows * topologies, data->rows * data->rows);
#endif

#pragma omp parallel for
  for (int r = 0; r < orders->rows; ++r) {
    int start = r / topologies;
    int *arr = new int[orders->cols];
    arr[0] = start;
    for (int i = 1; i < orders->cols; ++i) {
      arr[i] = i == start ? 0 : i;
    }
    shuffle_int(orders->cols - 1, arr + 1);
    for (int c = 0; c < orders->cols; ++c) {
      orders->inplace_set(r, c, arr[c]);
    }
    delete[] arr;
  }

  SMatrix consensus_network = std::make_shared<Matrix>(data->rows, data->rows);
  int cn_n_elements = consensus_network->rows * consensus_network->cols;

#pragma omp parallel for
  for (int o = 0; o < orders->rows; ++o) {
    SMatrix m_order = orders->list_elems_by_row_position(o + 1);
    std::vector<int> order = m_order->asVector<int>();
    SMatrix bnet = learn_struct_K2(data, sz, order, scoring_fn, max_parents);
    assert(consensus_network->rows == bnet->rows);
    assert(consensus_network->cols == bnet->cols);

#pragma omp critical
    for (int i = 0; i < cn_n_elements; ++i) {
      consensus_network->data[i] += bnet->data[i] ? 1 : 0;
    }

#if SAVE_NETWORKS
    for (int i = 0; i < cn_n_elements; ++i) {
      networks->data[i + cn_n_elements * o] = bnet->data[i];
    }
#endif
  }

  if (forkIndex == 0) {
    consensus_network->save(f_output);
#if SAVE_NETWORKS
    networks->save("networks.csv");
    orders->save("topologies.csv");
#endif
  }
  return 0;
}

int main(int argc, char **argv) {
  int forkIndex = 0, forkSize = 1;

  srand(time(NULL) ^ forkIndex);
  int threads = 1, topologies = 1, max_parents = 0;
  bool data_transposed = false;
  std::string data, output = "consensus.csv";
  std::string scoring_fn = "bayesian";
  int c;
  while ((c = getopt(argc, argv, "Thp:d:t:o:s:")) != -1) {
    switch (c) {
    case 'T': {
      data_transposed = true;
      break;
    }
    case 'p': {
      threads = atoi(optarg);
      assert(threads > 0);
      assert(threads <= omp_get_num_procs());
      break;
    }
    case 'm': {
      max_parents = atoi(optarg);
      assert(max_parents >= 0);
      break;
    }
    case 'd': {
      data = optarg;
      break;
    }
    case 't': {
      topologies = atoi(optarg);
      break;
    }
    case 'o': {
      output = optarg;
      break;
    }
    case 's': {
      scoring_fn = optarg;
      break;
    }
    case 'h':
    default: {
      puts(
          ": -p <num_threads> -d <data file> -t <topologies per gene> -o "
          "<output file> -m <max parents>");
      puts("~ -T (reads matrix transposed)");
      return 1;
    }
    }
  }
  if (data.size() < 1) {
    puts("You must send a data file using -d <file name>.");
    return 1;
  }
  omp_set_num_threads(threads);
  int status = exec(forkIndex, forkSize, data_transposed, data, topologies,
                    output, scoring_fn, max_parents);
  return status;
}
