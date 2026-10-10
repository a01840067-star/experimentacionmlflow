#!/bin/bash
python -m experimentacionmlflowModule.modeling.train                                  # base: lineal
python -m experimentacionmlflowModule.modeling.train --alpha 10                       # mas regularizacion
python -m experimentacionmlflowModule.modeling.train --degree 2 --alpha 1             # curvas
python -m experimentacionmlflowModule.modeling.train --degree 2 --alpha 10            # curvas + regularizacion
python -m experimentacionmlflowModule.modeling.train --degree 2 --alpha 1 --log-target
python -m experimentacionmlflowModule.modeling.train --degree 3 --alpha 0.01         