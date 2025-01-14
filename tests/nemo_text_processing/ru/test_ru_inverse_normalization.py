# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from parameterized import parameterized

from ..utils import CACHE_DIR, parse_test_case_file


class TestRuInverseNormalize:

    normalizer = InverseNormalizer(lang='ru', cache_dir=CACHE_DIR)

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_cardinal(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_ordinal(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    # @parameterized.expand(parse_test_case_file('ru_data_inverse_text_normalization/test_cases_ordinal_hard.txt'))
    # @pytest.mark.run_only_on('CPU')
    # @pytest.mark.unit
    # def test_denorm_ordinal_hard(self, test_input, expected):
    #     pred = self.normalizer.inverse_normalize(test_input, verbose=False)
    #     assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_decimal(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_electronic(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_date(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_measure.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_measure(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_money.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_money(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_time.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_time(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_whitelist.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_whitelist(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_word.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_word(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred
