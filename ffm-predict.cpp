#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffm.h"

using namespace std;
using namespace ffm;

struct Option {
  string test_path, model_path, output_path;
  ffm_float nds_rate;
};

string predict_help() {
  return string(
    "usage: ffm-predict [options] test_file model_file output_file\n"
    "\n"
    "options:\n"
    "-nds-rate: set the negative down sampling rate for unbalanced data\n"
  );
}

Option parse_option(int argc, char **argv) {
  vector<string> args;
  for (int i = 0; i < argc; i++)
    args.push_back(string(argv[i]));

  if (argc == 1)
    throw invalid_argument(predict_help());

  Option option;

  if (argc < 4 || argc > 6)
    throw invalid_argument("cannot parse argument");

  option.test_path = string(args[1]);
  option.model_path = string(args[2]);
  option.output_path = string(args[3]);
  ffm_int i = 1;
  for (; i < argc; i++) {
      if (args[i].compare("--nds-rate") == 0) {
        if (i == argc - 1)
          throw invalid_argument("need to specify nds_rate after --nds-rate");
        i++;
        option.nds_rate = atof(args[i].c_str());
        if (option.nds_rate <= 0)
          throw invalid_argument("number of nds_rate should be greater than zero");
    }
  }

  return option;
}

void predict(string test_path, string model_path, string output_path, ffm_float nds_rate = 1.0) {
  int const kMaxLineSize = 1000000;

  FILE *f_in = fopen(test_path.c_str(), "r");
  ofstream f_out(output_path);
  char line[kMaxLineSize];

  ffm_model *model = ffm_load_model(model_path.c_str());

  ffm_double loss = 0;
  vector<ffm_node> x;
  ffm_int i = 0;

  for (; fgets(line, kMaxLineSize, f_in) != nullptr; i++) {
    x.clear();
    char *y_char = strtok(line, " \t");
    ffm_float y = (atoi(y_char) > 0) ? 1.0f : -1.0f;

    while (true) {
      char *field_char = strtok(nullptr, ":");
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, " \t");
      if (field_char == nullptr || *field_char == '\n')
        break;

      ffm_node N;
      N.f = atoi(field_char);
      N.j = atoi(idx_char);
      N.v = atof(value_char);

      x.push_back(N);
    }

    ffm_float y_bar = ffm_predict(x.data(), x.data() + x.size(), model, nds_rate);

    loss -= y == 1 ? log(y_bar) : log(1 - y_bar);

    f_out << y << "," << y_bar << "\n";
  }

  loss /= i;

  cout << "logloss = " << fixed << setprecision(5) << loss << endl;

  ffm_destroy_model(&model);
}

int main(int argc, char **argv) {
  Option option;
  try {
    option = parse_option(argc, argv);
  } catch (invalid_argument const &e) {
    cout << e.what() << endl;
    return 1;
  }

  predict(option.test_path, option.model_path, option.output_path, option.nds_rate);
}
