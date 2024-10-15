import requests
import json


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


class CommonOcr(object):
    def __init__(self, imagepath):
        # 请登录后前往 “工作台-账号设置-开发者信息” 查看 x-ti-app-id
        # 示例代码中 x-ti-app-id 非真实数据
        self._app_id = '374c737cff005324cf43396e3984c9a7'
        # 请登录后前往 “工作台-账号设置-开发者信息” 查看 x-ti-secret-code
        # 示例代码中 x-ti-secret-code 非真实数据
        self._secret_code = '3b773612518dab6f5d987dc65bce35e1'
        self.image = get_file_content(imagepath)

    def recognize(self):
        # 通用表格识别
        url = 'https://api.textin.com/ai/service/v2/recognize/table/multipage'
        head = {}
        try:
            annotation = []
            image = self.image
            head['x-ti-app-id'] = self._app_id
            head['x-ti-secret-code'] = self._secret_code
            result = requests.post(url, data=image, headers=head)
            print(result.text)
            for i in json.loads(result.text)["result"]["pages"][0]["tables"]:
                if "table_cells" not in i.keys():
                    continue
                for j in i["table_cells"]:
                    for k in j["lines"]:
                        annotation.append([k["text"], k["handwritten"], k["position"], k["score"]])
            return annotation
        except Exception as e:
            return e


if __name__ == "__main__":
    response = CommonOcr(r'test.png')
    print(response.recognize())
