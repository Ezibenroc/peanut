if [ $# -ne 2 ] ; then
    echo $#
    echo "Syntax: $0 <frontend> <username>"
    exit 1
fi
frontend=$1
user=$2

rm -rf dist
python setup.py bdist_wheel
scp dist/*whl $frontend.g5k:/home/$user
ssh $frontend.g5k 'python3 -m pip uninstall -y peanut ; python3 -m pip install peanut*whl --user'
