"""Package the notebook and one Section A PDF; default PDF is a typed study guide."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--section-a',type=Path,default=ROOT/'section_a'/'SM24-01-SectionA_Worked_Answers_Cai_Haochen_58561440.pdf')
args=parser.parse_args()
notebook=ROOT/'submission'/'Assign01_Cai_Haochen_58561440.ipynb'
assert notebook.is_file()
assert args.section_a.is_file() and args.section_a.read_bytes().startswith(b'%PDF-')
target=ROOT/'submission'/'Assign01_Cai_Haochen_58561440.zip'
with ZipFile(target,'w',ZIP_DEFLATED) as archive:
    archive.write(notebook,notebook.name)
    archive.write(args.section_a,args.section_a.name)
print(target)
print('Verify Section A is your genuine handwritten scan before submission; packaging does not verify handwriting.')
