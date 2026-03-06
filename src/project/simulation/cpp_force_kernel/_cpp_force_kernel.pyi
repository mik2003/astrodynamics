# _cpp_force_kernel.pyi

# SPDX-FileCopyrightText: © 2026 Michelangelo Secondo <michelangelo@secondo.aero>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from project.utils import FloatArray

def point_mass_cpp(
    state: FloatArray,
    mu: FloatArray,
    out: FloatArray,
) -> None: ...
