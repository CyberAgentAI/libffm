#pragma GCC diagnostic ignored "-Wunused-result"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <pmmintrin.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined USEOMP
#include <omp.h>
#endif

#include "ffm.h"

namespace ffm {

namespace {

using namespace std;

ffm_int const kALIGNByte = 16;
ffm_int const kALIGN = kALIGNByte / sizeof(ffm_float);
ffm_int const kMaxLineSize = 100000;

inline ffm_float wTx(ffm_node *begin, ffm_node *end, ffm_float r,
                     ffm_model &model, ffm_float iw = 1.0f, ffm_float kappa = 0,
                     ffm_float eta = 0, ffm_float lambda = 0,
                     bool do_update = false) {
  ffm_long align0 = (ffm_long)model.k * 2;
  ffm_long align1 = (ffm_long)model.m * align0;

  __m128 XMMkappa = _mm_set1_ps(kappa);
  __m128 XMMeta = _mm_set1_ps(eta);
  __m128 XMMlambda = _mm_set1_ps(lambda);

  __m128 XMMt = _mm_setzero_ps();

  for (ffm_node *N1 = begin; N1 != end; N1++) {
    ffm_int j1 = N1->j;
    ffm_int f1 = N1->f;
    ffm_float v1 = N1->v;
    if (j1 >= model.n || f1 >= model.m)
      continue;

    for (ffm_node *N2 = N1 + 1; N2 != end; N2++) {
      ffm_int j2 = N2->j;
      ffm_int f2 = N2->f;
      ffm_float v2 = N2->v;
      if (j2 >= model.n || f2 >= model.m)
        continue;

      ffm_float *w1 = model.W + j1 * align1 + f2 * align0;
      ffm_float *w2 = model.W + j2 * align1 + f1 * align0;

      __m128 XMMv = _mm_set1_ps(v1 * v2 * r);

      if (do_update) {
        __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

        ffm_float *wg1 = w1 + model.k;
        ffm_float *wg2 = w2 + model.k;
        for (ffm_int d = 0; d < model.k; d += 4) {
          __m128 XMMw1 = _mm_load_ps(w1 + d);
          __m128 XMMw2 = _mm_load_ps(w2 + d);

          __m128 XMMwg1 = _mm_load_ps(wg1 + d);
          __m128 XMMwg2 = _mm_load_ps(wg2 + d);

          __m128 XMMg1 = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw1),
                                    _mm_mul_ps(XMMkappav, XMMw2));
          __m128 XMMg2 = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw2),
                                    _mm_mul_ps(XMMkappav, XMMw1));

          XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
          XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

          XMMw1 = _mm_sub_ps(
              XMMw1,
              _mm_mul_ps(XMMeta, _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
          XMMw2 = _mm_sub_ps(
              XMMw2,
              _mm_mul_ps(XMMeta, _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

          _mm_store_ps(w1 + d, XMMw1);
          _mm_store_ps(w2 + d, XMMw2);

          _mm_store_ps(wg1 + d, XMMwg1);
          _mm_store_ps(wg2 + d, XMMwg2);
        }
      } else {
        for (ffm_int d = 0; d < model.k; d += 4) {
          __m128 XMMw1 = _mm_load_ps(w1 + d);
          __m128 XMMw2 = _mm_load_ps(w2 + d);

          XMMt = _mm_add_ps(XMMt, _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
        }
      }
    }
  }

  if (do_update)
    return 0;

  XMMt = _mm_hadd_ps(XMMt, XMMt);
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  ffm_float t;
  _mm_store_ss(&t, XMMt);

  return t;
}

ffm_float *malloc_aligned_float(ffm_long size) {
  void *ptr;

#ifdef _WIN32
  ptr = _aligned_malloc(size * sizeof(ffm_float), kALIGNByte);
  if (ptr == nullptr)
    throw bad_alloc();
#else
  int status = posix_memalign(&ptr, kALIGNByte, size * sizeof(ffm_float));
  if (status != 0)
    throw bad_alloc();
#endif

  return (ffm_float *)ptr;
}

ffm_model *init_model(ffm_int n, ffm_int m, ffm_parameter param) {
  ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k / kALIGN) * kALIGN;

  ffm_model *model = new ffm_model;
  model->n = n;
  model->k = k_aligned;
  model->m = m;
  model->W = nullptr;
  model->normalization = param.normalization;

  try {
    model->W = malloc_aligned_float((ffm_long)n * m * k_aligned * 2);
  } catch (bad_alloc const &e) {
    ffm_destroy_model(&model);
    throw;
  }

  ffm_float coef = 1.0f / sqrt(param.k);
  ffm_float *w = model->W;

  default_random_engine generator;
  uniform_real_distribution<ffm_float> distribution(0.0, 1.0);

  for (ffm_int j = 0; j < model->n; j++) {
    for (ffm_int f = 0; f < model->m; f++) {
      for (ffm_int d = 0; d < param.k; d++, w++)
        *w = coef * distribution(generator);
      for (ffm_int d = param.k; d < k_aligned; d++, w++)
        *w = 0;
      for (ffm_int d = k_aligned; d < 2 * k_aligned; d++, w++)
        *w = 1;
    }
  }

  return model;
}

void shrink_model(ffm_model &model, ffm_int k_new) {
  for (ffm_int j = 0; j < model.n; j++) {
    for (ffm_int f = 0; f < model.m; f++) {
      ffm_float *src = model.W + ((ffm_long)j * model.m + f) * model.k * 2;
      ffm_float *dst = model.W + ((ffm_long)j * model.m + f) * k_new;
      copy(src, src + k_new, dst);
    }
  }

  model.k = k_new;
}

vector<ffm_float> normalize(ffm_problem &prob) {
  vector<ffm_float> R(prob.l);
#if defined USEOMP
#pragma omp parallel for schedule(static)
#endif
  for (ffm_int i = 0; i < prob.l; i++) {
    ffm_float norm = 0;
    for (ffm_long p = prob.P[i]; p < prob.P[i + 1]; p++)
      norm += prob.X[p].v * prob.X[p].v;
    R[i] = 1 / norm;
  }

  return R;
}

shared_ptr<ffm_model> train(ffm_problem *tr, vector<ffm_int> &order,
                            ffm_parameter param, ffm_problem *va = nullptr,
                            ffm_importance_weights *iws = nullptr,
                            ffm_importance_weights *iwvs = nullptr) {
#if defined USEOMP
  ffm_int old_nr_threads = omp_get_num_threads();
  omp_set_num_threads(param.nr_threads);
#endif

  shared_ptr<ffm_model> model =
      shared_ptr<ffm_model>(init_model(tr->n, tr->m, param),
                            [](ffm_model *ptr) { ffm_destroy_model(&ptr); });

  vector<ffm_float> R_tr, R_va;
  if (param.normalization) {
    R_tr = normalize(*tr);
    if (va != nullptr)
      R_va = normalize(*va);
  } else {
    R_tr = vector<ffm_float>(tr->l, 1);
    if (va != nullptr)
      R_va = vector<ffm_float>(va->l, 1);
  }

  bool auto_stop = param.auto_stop && va != nullptr && va->l != 0;

  ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k / kALIGN) * kALIGN;
  ffm_long w_size = (ffm_long)model->n * model->m * k_aligned * 2;
  vector<ffm_float> prev_W;
  if (auto_stop)
    prev_W.assign(w_size, 0);
  ffm_double best_va_loss = numeric_limits<ffm_double>::max();
  ffm_int best_iteration = 0;
  ffm_int loss_worse_counter = param.auto_stop_threshold;

  if (!param.quiet) {
    if (param.auto_stop && (va == nullptr || va->l == 0))
      cerr << "warning: ignoring auto-stop because there is no validation set"
           << endl;

    cout.width(4);
    cout << "iter";
    cout.width(13);
    cout << "tr_logloss";
    if (va != nullptr && va->l != 0) {
      cout.width(13);
      cout << "va_logloss";
    }
    cout << endl;
  }

  for (ffm_int iter = 1; iter <= param.nr_iters; iter++) {
    ffm_double tr_loss = 0;
    if (param.random)
      random_shuffle(order.begin(), order.end());
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+ : tr_loss)
#endif
    for (ffm_int ii = 0; ii < tr->l; ii++) {
      ffm_int i = order[ii];

      ffm_float y = tr->Y[i];
      ffm_float iw = 1.0;
      if (iws != nullptr) {
        iw = iws->W[i];
      }

      ffm_node *begin = &tr->X[tr->P[i]];

      ffm_node *end = &tr->X[tr->P[i + 1]];

      ffm_float r = R_tr[i];

      ffm_float t = wTx(begin, end, r, *model);

      ffm_float expnyt = exp(-y * t);

      tr_loss += log(1 + expnyt);

      ffm_float kappa = -y * expnyt / (1 + expnyt);

      wTx(begin, end, r, *model, iw, kappa, param.eta, param.lambda, true);
    }

    if (!param.quiet) {
      tr_loss /= tr->l;

      cout.width(4);
      cout << iter;
      cout.width(13);
      cout << fixed << setprecision(5) << tr_loss;
      if (va != nullptr && va->l != 0) {
        ffm_double va_loss = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+ : va_loss)
#endif
        for (ffm_int i = 0; i < va->l; i++) {
          ffm_float y = va->Y[i];
          ffm_float iwv;

          if (iwvs == nullptr) {
            iwv = 1;
          } else {
            iwv = iwvs->W[i];
          }

          ffm_node *begin = &va->X[va->P[i]];

          ffm_node *end = &va->X[va->P[i + 1]];

          ffm_float r = R_va[i];

          ffm_float t = wTx(begin, end, r, *model);

          ffm_float expnyt = exp(-y * t);

          va_loss += log(1 + expnyt) * iwv;
        }

        if (iwvs == nullptr) {
          va_loss /= va->l;
        } else {
          va_loss /= iwvs->sum;
        }

        cout.width(13);
        cout << fixed << setprecision(5) << va_loss;

        if (auto_stop) {
          if (va_loss > best_va_loss) {
            loss_worse_counter--;
            if (loss_worse_counter <= 0) {
              memcpy(model->W, prev_W.data(), w_size * sizeof(ffm_float));
              cout << endl
                   << "Auto-stop. Use model at " << best_iteration
                   << "th iteration." << endl;
              break;
            } else {
              memcpy(prev_W.data(), model->W, w_size * sizeof(ffm_float));
            }
          } else {
            loss_worse_counter = param.auto_stop_threshold; // reset the counter
            memcpy(prev_W.data(), model->W, w_size * sizeof(ffm_float));
            best_va_loss = va_loss;
            best_iteration = iter;
          }
        }
      }
      cout << endl;
    }
  }

  // generate json meta file.
  if (param.json_meta_path != nullptr) {
    ofstream f_out(param.json_meta_path);
    if (f_out.is_open()) {
      f_out << "{\"best_iteration\": " << best_iteration << ", \"best_va_loss\": " << best_va_loss << "}\n" << flush;
      f_out.close();
    }
  }

  shrink_model(*model, param.k);

#if defined USEOMP
  omp_set_num_threads(old_nr_threads);
#endif

  return model;
}

} // unnamed namespace

ffm_problem *ffm_read_problem(char const *path) {
  if (strlen(path) == 0)
    return nullptr;

  FILE *f = fopen(path, "r");
  if (f == nullptr)
    return nullptr;

  ffm_problem *prob = new ffm_problem;
  prob->l = 0;
  prob->n = 0;
  prob->m = 0;
  prob->X = nullptr;
  prob->P = nullptr;
  prob->Y = nullptr;

  char line[kMaxLineSize];

  ffm_long nnz = 0;
  for (; fgets(line, kMaxLineSize, f) != nullptr; prob->l++) {
    strtok(line, " \t");
    for (;; nnz++) {
      char *ptr = strtok(nullptr, " \t");
      if (ptr == nullptr || *ptr == '\n')
        break;
    }
  }
  rewind(f);

  prob->X = new ffm_node[nnz];
  prob->P = new ffm_long[prob->l + 1];
  prob->Y = new ffm_float[prob->l];

  ffm_long p = 0;
  prob->P[0] = 0;
  for (ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++) {
    char *y_char = strtok(line, " \t");
    ffm_float y = (atoi(y_char) > 0) ? 1.0f : -1.0f;
    prob->Y[i] = y;

    for (;; p++) {
      char *field_char = strtok(nullptr, ":");
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, " \t");
      if (field_char == nullptr || *field_char == '\n')
        break;

      ffm_int field = atoi(field_char);
      ffm_int idx = atoi(idx_char);
      ffm_float value = atof(value_char);

      prob->m = max(prob->m, field + 1);
      prob->n = max(prob->n, idx + 1);

      prob->X[p].f = field;
      prob->X[p].j = idx;
      prob->X[p].v = value;
    }
    prob->P[i + 1] = p;
  }

  fclose(f);

  return prob;
}

ffm_importance_weights *ffm_read_importance_weights(char const *path) {
  if (strlen(path) == 0)
    return nullptr;

  FILE *f = fopen(path, "r");
  if (f == nullptr)
    return nullptr;

  ffm_importance_weights *weights = new ffm_importance_weights;
  weights->l = 0;
  weights->sum = 0;
  weights->W = nullptr;

  char line[kMaxLineSize];

  ffm_long nnz = 0;
  for (; fgets(line, kMaxLineSize, f) != nullptr; weights->l++) {
    strtok(line, " \t");
    for (;; nnz++) {
      char *ptr = strtok(nullptr, " \t");
      if (ptr == nullptr || *ptr == '\n')
        break;
    }
  }
  rewind(f);

  weights->W = new ffm_float[weights->l];

  for (ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++) {
    ffm_float iw = (ffm_float)atof(line);
    weights->sum += iw;
    weights->W[i] = iw;
  }
  fclose(f);
  return weights;
}

void ffm_destroy_problem(ffm_problem **prob) {
  if (prob == nullptr || *prob == nullptr)
    return;
  delete[](*prob)->X;
  delete[](*prob)->P;
  delete[](*prob)->Y;
  delete *prob;
  *prob = nullptr;
}

ffm_int ffm_save_model(ffm_model *model, char const *path) {
  ofstream f_out(path);
  if (!f_out.is_open())
    return 1;

  f_out << "n " << model->n << "\n";
  f_out << "m " << model->m << "\n";
  f_out << "k " << model->k << "\n";
  f_out << "normalization " << model->normalization << "\n";

  ffm_float *ptr = model->W;
  for (ffm_int j = 0; j < model->n; j++) {
    for (ffm_int f = 0; f < model->m; f++) {
      f_out << "w" << j << "," << f << " ";
      for (ffm_int d = 0; d < model->k; d++, ptr++)
        f_out << *ptr << " ";
      f_out << "\n";
    }
  }

  return 0;
}

ffm_int ffm_save_production_model(ffm_model *model, char const *path,
                                  char const *key_prefix) {
  ofstream f_out(path);
  if (!f_out.is_open())
    return 1;

  ffm_float *ptr = model->W;
  for (ffm_int j = 0; j < model->n; j++) {
    if (strcmp(key_prefix, "") != 0)
      f_out << "{\"key\":\"" << key_prefix << "_" << j << "\",\"value\":{";
    else
      f_out << "{\"key\":\"" << j << "\",\"value\":{";

    for (ffm_int f = 0; f < model->m; f++) {
      f_out << "\"" << f << "\":[";
      for (ffm_int d = 0; d < model->k; d++, ptr++) {
        if (d == model->k - 1) {
          if (f == model->m - 1) {
            f_out << *ptr << "]";
          } else {
            f_out << *ptr << "],";
          }
        } else {
          f_out << *ptr << ",";
        }
      }
    }
    f_out << "}}\n";
  }

  return 0;
}

ffm_model *ffm_load_model(char const *path) {
  ifstream f_in(path);
  if (!f_in.is_open())
    return nullptr;

  string dummy;

  ffm_model *model = new ffm_model;
  model->W = nullptr;

  f_in >> dummy >> model->n >> dummy >> model->m >> dummy >> model->k >>
      dummy >> model->normalization;

  try {
    model->W = malloc_aligned_float((ffm_long)model->m * model->n * model->k);
  } catch (bad_alloc const &e) {
    ffm_destroy_model(&model);
    return nullptr;
  }

  ffm_float *ptr = model->W;
  for (ffm_int j = 0; j < model->n; j++) {
    for (ffm_int f = 0; f < model->m; f++) {
      f_in >> dummy;
      for (ffm_int d = 0; d < model->k; d++, ptr++)
        f_in >> *ptr;
    }
  }

  return model;
}

void ffm_destroy_model(ffm_model **model) {
  if (model == nullptr || *model == nullptr)
    return;
#ifdef _WIN32
  _aligned_free((*model)->W);
#else
  free((*model)->W);
#endif
  delete *model;
  *model = nullptr;
}

ffm_parameter ffm_get_default_param() {
  ffm_parameter param;

  param.eta = 0.2;
  param.lambda = 0.00002;
  param.nr_iters = 15;
  param.k = 4;
  param.nr_threads = 1;
  param.auto_stop_threshold = -1;
  param.quiet = false;
  param.normalization = true;
  param.random = true;
  param.auto_stop = false;
  param.json_meta_path = nullptr;

  return param;
}

ffm_model *ffm_train_with_validation(ffm_problem *tr, ffm_problem *va,
                                     ffm_importance_weights *iws,
                                     ffm_importance_weights *iwvs,
                                     ffm_parameter param) {
  vector<ffm_int> order(tr->l);
  for (ffm_int i = 0; i < tr->l; i++)
    order[i] = i;

  shared_ptr<ffm_model> model = train(tr, order, param, va, iws, iwvs);

  ffm_model *model_ret = new ffm_model;

  model_ret->n = model->n;
  model_ret->m = model->m;
  model_ret->k = model->k;
  model_ret->normalization = model->normalization;

  model_ret->W = model->W;
  model->W = nullptr;

  return model_ret;
}

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model *model) {
  ffm_float r = 1;
  if (model->normalization) {
    r = 0;
    for (ffm_node *N = begin; N != end; N++)
      r += N->v * N->v;
    r = 1 / r;
  }

  ffm_long align0 = (ffm_long)model->k;
  ffm_long align1 = (ffm_long)model->m * align0;

  ffm_float t = 0;
  for (ffm_node *N1 = begin; N1 != end; N1++) {
    ffm_int j1 = N1->j;
    ffm_int f1 = N1->f;
    ffm_float v1 = N1->v;
    if (j1 >= model->n || f1 >= model->m)
      continue;

    for (ffm_node *N2 = N1 + 1; N2 != end; N2++) {
      ffm_int j2 = N2->j;
      ffm_int f2 = N2->f;
      ffm_float v2 = N2->v;
      if (j2 >= model->n || f2 >= model->m)
        continue;

      ffm_float *w1 = model->W + j1 * align1 + f2 * align0;
      ffm_float *w2 = model->W + j2 * align1 + f1 * align0;

      ffm_float v = v1 * v2 * r;

      for (ffm_int d = 0; d < model->k; d++)
        t += w1[d] * w2[d] * v;
    }
  }

  return 1 / (1 + exp(-t));
}

ffm_float ffm_cross_validation(ffm_problem *prob, ffm_int nr_folds,
                               ffm_parameter param) {
#if defined USEOMP
  ffm_int old_nr_threads = omp_get_num_threads();
  omp_set_num_threads(param.nr_threads);
#endif

  bool quiet = param.quiet;
  param.quiet = true;

  vector<ffm_int> order(prob->l);
  for (ffm_int i = 0; i < prob->l; i++)
    order[i] = i;
  random_shuffle(order.begin(), order.end());

  if (!quiet) {
    cout.width(4);
    cout << "fold";
    cout.width(13);
    cout << "logloss";
    cout << endl;
  }

  ffm_double loss = 0;
  ffm_int nr_instance_per_fold = prob->l / nr_folds;
  for (ffm_int fold = 0; fold < nr_folds; fold++) {
    ffm_int begin = fold * nr_instance_per_fold;
    ffm_int end = min(begin + nr_instance_per_fold, prob->l);

    vector<ffm_int> order1;
    for (ffm_int i = 0; i < begin; i++)
      order1.push_back(order[i]);
    for (ffm_int i = end; i < prob->l; i++)
      order1.push_back(order[i]);

    shared_ptr<ffm_model> model = train(prob, order1, param);

    ffm_double loss1 = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+ : loss1)
#endif
    for (ffm_int ii = begin; ii < end; ii++) {
      ffm_int i = order[ii];

      ffm_float y = prob->Y[i];

      ffm_node *begin = &prob->X[prob->P[i]];

      ffm_node *end = &prob->X[prob->P[i + 1]];

      ffm_float y_bar = ffm_predict(begin, end, model.get());

      loss1 -= y == 1 ? log(y_bar) : log(1 - y_bar);
    }
    loss += loss1;

    if (!quiet) {
      cout.width(4);
      cout << fold;
      cout.width(13);
      cout << fixed << setprecision(4) << loss1 / (end - begin);
      cout << endl;
    }
  }

  if (!quiet) {
    cout.width(17);
    cout.fill('=');
    cout << "" << endl;
    cout.fill(' ');
    cout.width(4);
    cout << "avg";
    cout.width(13);
    cout << fixed << setprecision(4) << loss / prob->l;
    cout << endl;
  }

#if defined USEOMP
  omp_set_num_threads(old_nr_threads);
#endif

  return loss / prob->l;
}

} // namespace ffm
