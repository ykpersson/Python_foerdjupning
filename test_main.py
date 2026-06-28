import logger
import setting

from price_builder import PriceBuilderV3
from index_builder import IndexBuilder
from tech_builder import TechBuilder
from price_combinatorics_builder import PriceCombinatoricsBuilder
from band_builder import BandBuilder
from target2 import TargetBuilder
from dataset_builder import DatasetBuilder


def main():
    logger.init(setting.LOG_DIR)

    print("\n=== 1. PRICE ===")
    PriceBuilderV3().build_all()

    print("\n=== 2. INDEX ===")
    IndexBuilder().build_all()

    print("\n=== 3. TECH ===")
    TechBuilder().build_all()

    print("\n=== 4. PRICE COMBINATORICS ===")
    PriceCombinatoricsBuilder().build()

    print("\n=== 5. BAND BUILDER (G1–G4) ===")
    BandBuilder().build_all()

    print("\n=== 6. TARGET BUILDER (lokal per band, ordinal) ===")
    TargetBuilder().build_all()

    print("\n=== 7. DATASET BUILDER ===")
    DatasetBuilder().build_all()

    print("\n=== KLART ===")


if __name__ == "__main__":
    main()
