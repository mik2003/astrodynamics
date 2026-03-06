# SPDX-FileCopyrightText: © 2026 Michelangelo Secondo <michelangelo@secondo.aero>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the profiling viewer as a module: python -m profiling"""

from .profile_stats import cli

if __name__ == "__main__":
    cli()
