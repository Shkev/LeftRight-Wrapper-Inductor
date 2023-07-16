import json

from supervised_single_entry_lr_wrapper import SingleEntryLRWrapper

def main():
    with open('labelled_websites.json') as f:
        data = json.load(f)
        file_paths = [entry['file_path'] for entry in data]
        labels = [[tuple(label[1]) for label in entry['labels']] for entry in data]
        label_names = [label[0] for label in data[0]['labels']]
    wrapper = SingleEntryLRWrapper(file_paths, labels)
    # print(wrapper.left)
    # print(wrapper.right)
    
    # testing on training inputs
    print("\nTesting on training inputs...")
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            P = f.read()
        extracted_attr_list = wrapper.execLR(P)
        extracted_attr_dict = {l : attr for l, attr in zip(label_names, extracted_attr_list)}
        print(extracted_attr_dict)

    # testing on new input
    print("\nTesting on new input...")
    path = 'websites/aleksei_aksimentiev_physics_uiuc.html'
    with open(path, 'r', encoding='utf-8') as f:
        P = f.read()
    extracted_attr_list = wrapper.execLR(P)
    extracted_attr_dict = {l : attr for l, attr in zip(label_names, extracted_attr_list)}
    print(extracted_attr_dict)

if __name__ == '__main__':
    main()
