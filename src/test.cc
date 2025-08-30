#include <util/parse.h>
#include <util/search.h>

void assert(bool condition, std::string str = "") {
  if (!condition) {
    throw std::runtime_error{str};
  }
}

void assert_search_value(std::string str, float value, float tol = .01) {
  const auto [battle, duration] = Parse::parse_battle(str);
  RuntimeSearch::run()
}

namespace Search {

void test_confusion() {
  std::string foo = "snorlax 1% swift| jolteon swift 1% (conf:1)";
}

} // namespace Search