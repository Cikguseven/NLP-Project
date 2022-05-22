allowed_extensions = {'json', 'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


print(allowed_file('p.csv'))