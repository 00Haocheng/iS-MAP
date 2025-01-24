# iSMAP is a modified version of https://github.com/idiap/ESLAM
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2023 Johari, M. M. and Carta, C. and Fleuret, F.
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
import argparse

from src import config
from src.iSMAP import iSMAP
import time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/iSMAP.yaml')
    ismap = iSMAP(cfg, args)
    ismap.run()


if __name__ == '__main__':
    main()
