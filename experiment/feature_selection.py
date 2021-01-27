import os
import json
from tempfile import mkdtemp
from shutil import rmtree
import java
with java:
    from python import FeatureSelector


class FSelector(object):

    def __init__(self, df):

        self.df = df

        from jnius import JavaException
        self.temp_dir = mkdtemp()
        file_path = self.temp_dir + '/my-feature'
        real_file_path = dump_dataframe(file_path, self.df)
        # time.sleep(2)
        try:
            self.feature_select(path=real_file_path)
        except JavaException as e:
            print('#' * 80)
            for stack_frame in e.stacktrace:
                print(stack_frame)
            raise
        finally:
            rmtree(self.temp_dir)

    def feature_select(self, path):

        fs = FeatureSelector(path)
        result = fs.buildFeatureSelector(fs)

        self.all_idx = result

        # return result


def dump_dataframe(path, df):
    'Save an dataframe in three files: data.csv, attr.txt and name.nfo.'
    print(path)
    # data_file = new_path(path, 'csv')
    data_file = path + '.csv'
    df.to_csv(data_file, header=True, index=False)
    return data_file


def new_path(path: str, ext='') -> str:
    'Return the original path, but with a different suffix name.'
    path = list(os.path.splitext(path))
    path[1] = ext
    return ''.join(path)


def get_feature_idx(index, all_index):
    # json_str = json.loads(all_index.replace('\"', '\''))
    dict_str = json.loads(all_index)
    return dict_str[index]["featureIndex"]


if __name__ == '__main__':

    pass