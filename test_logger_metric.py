import logger_metric
from enum import IntEnum

class TestEnum(IntEnum):
    CLASS1=0
    CLASS2=2
    CLASS3=5

    def __str__(self):
        return self.name.lower()

def test_get_errors():
    error_list = [10.0,30.0,60.0,80.0,90.0,200.0,200.0]
    class_list = [0,0,5,5,2,2,2]
    hard = [0,0,0,0,0,1,1]
    easy_stats, all_stats = logger_metric.get_errors(error_list, class_list, hard, TestEnum)
    logger_metric.print_stats(easy_stats)
    logger_metric.print_stats(easy_stats, verbose=True)


def main():
    test_get_errors()

if __name__ == '__main__':
    main()
