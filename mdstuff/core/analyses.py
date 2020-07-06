import abc


class Analysis(abc.ABC):
    """
    Base class for all analyses.
    """

    @abc.abstractmethod
    def save(self, filename: str):
        """
        Save the results.

        Args:
            filename (str): The file name.
        """
        pass


class ParallelAnalysis(Analysis):
    """
    Base class for analyses that can run in parallel.
    """

    @abc.abstractmethod
    def update(self):
        """
        Update the analysis in each step.
        """
        pass


class SerialAnalysis(Analysis):
    """
    Base class for analyses that can not run in parallel.
    """

    @abc.abstractmethod
    def run(self, universe):
        """
        Run the analysis.
        """
        pass
