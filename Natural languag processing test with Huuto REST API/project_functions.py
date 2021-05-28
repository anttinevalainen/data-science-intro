import json
import pandas as pd
import math

def getallrows(categorynumber):
    '''
    Get all the items of a single category to a dataframe
    from huuto.net REST API.

    Goes through huuto.net API and page by page appends
    items to a single pandas dataframe

    Args:
        - categorynumber : int
            number of the category to get the dataframe from
            see categories and subcategories from https://api.huuto.net/1.1/categories/

    Returns:
        - df : Pandas DataFrame
            Dataframe object of all the listed items in the given category
    '''
    assert type(categorynumber) == int, 'categoryNumber must be an integer'

    columns = ['links','id','title','category','seller',
               'sellerId','currentPrice','buyNowPrice','saleMethod','listTime',
               'postalCode','location','closingTime','bidderCount','offerCount',
               'hasReservePrice','hasReservePriceExceeded','upgrades','images']

    df = pd.DataFrame(columns = columns)

    fp = 'https://api.huuto.net/1.1/categories/' + str(categorynumber) + '/items'

    data = pd.read_json(path_or_buf = fp, orient = 'index').transpose()

    pagecounter = data['totalCount'][0] / 50
    pagecount = math.ceil(pagecounter)

    for i in range(1, (pagecount + 1)):

        fp = ('https://api.huuto.net/1.1/categories/459/items?page=' +
              str(i) + '&category=' + str(categorynumber))

        data2 = pd.read_json(path_or_buf = fp, orient = 'index').transpose()

        newrows = data2['items'][0]

        df = df.append(newrows, ignore_index=True)

    df['closed'] = 0
    return df

def getclosed(categorynumber):
    '''
    Get all the closed items of a single category to a dataframe
    from huuto.net REST API.

    Goes through huuto.net API and page by page appends
    items to a single pandas dataframe

    Args:
        - categorynumber : int
            number of the category to get the dataframe from
            see categories and subcategories
            from https://api.huuto.net/1.1/categories/

    Returns:
        - df : Pandas DataFrame
            Dataframe object of all the listed items in
            the given category
    '''
    assert type(categorynumber) == int, 'categoryNumber must be an integer'

    columns = ['links','id','title','category','seller',
               'sellerId','currentPrice','buyNowPrice','saleMethod','listTime',
               'postalCode','location','closingTime','bidderCount','offerCount',
               'hasReservePrice','hasReservePriceExceeded','upgrades','images']

    df = pd.DataFrame(columns = columns)

    fp = 'https://api.huuto.net/1.1/categories/' + str(categorynumber) + '/items/closed'

    data = pd.read_json(path_or_buf = fp, orient = 'index').transpose()

    pagecounter = data['totalCount'][0] / 50
    pagecount = math.ceil(pagecounter)

    for i in range(1, (pagecount + 1)):

        fp = ('https://api.huuto.net/1.1/categories/459/items/closed?status=closed&page=' +
              str(i) + '&category=' + str(categorynumber))

        data2 = pd.read_json(path_or_buf = fp, orient = 'index').transpose()

        newrows = data2['items'][0]

        df = df.append(newrows, ignore_index=True)

    df['closed'] = 1
    return df