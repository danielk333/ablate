import pytest
import shutil



@pytest.hookimpl(hookwrapper=True)
def pytest_collection_finish(session):
    '''Wrapper for pytest that ensures that if pytest is run as a developer directly in the repository, any __pycache__ files create do not interfere with subsequent pip install commands.
    '''

    # will execute before tests

    outcome = yield

    # will execute after all non-hookwrappers executed
    
    shutil.rmtree(
        session.startdir + '/tests/__pycache__',
        ignore_errors = True,
    )
