"""Verify data, isolation from true labels, probability validation, and metrics.
These tests do not simulate model scores or validate model accuracy.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor/open-jev'))
import pipeline as p
from jev.api import compile_request, candidate_prompts, format_response

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((HERE / 'settings.json').read_text())
        self.labels = list(self.settings['categories'])

    def test_label_isolation_and_real_api_schema(self):
        request = p.category_request('A bank announces earnings', self.settings)
        records = compile_request(request['state'], request['questions'])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['answer_keys'], self.labels)
        self.assertNotIn('true_label', json.dumps(request))
        self.assertEqual(len(candidate_prompts(records[0])), 3)
        # Fixed distribution is a software contract fixture, not model inference.
        response = format_response(records, [[0.1, 0.2, 0.7]])
        probs, selected = p.validate_probabilities(response['answers']['topic'], self.labels)
        self.assertEqual(selected, 'finance')
        self.assertEqual(probs['finance'], 0.7)

    def test_invalid_distributions_rejected(self):
        for vals in ([.1,.1,.1], [float('nan'), .2,.8], [-.1,.2,.9]):
            with self.assertRaises(ValueError):
                p.validate_probabilities({'probabilities': dict(zip(self.labels, vals)), 'choice': 'finance'}, self.labels)

    def test_metrics_known_answer(self):
        rows=[]
        for truth, pred, probs in [('sports','sports',[.8,.1,.1]), ('politics','finance',[.1,.2,.7])]:
            rows.append({'status':'ok','true_label':truth,'predicted_label':pred,'correct':truth==pred,
                         **dict(zip(['prob_'+l for l in self.labels],probs))})
        rows.append({'status':'error','true_label':'finance','predicted_label':'','correct':''})
        m=p.evaluate(rows,self.labels)
        self.assertEqual(m['accuracy'],.5)
        self.assertAlmostEqual(m['multiclass_brier'],.6)
        self.assertAlmostEqual(m['log_loss'],-(__import__('math').log(.8)+__import__('math').log(.2))/2)
        self.assertEqual(m['failed_rows'],1)
        self.assertEqual(m['confusion_matrix']['politics']['finance'],1)

    def test_audit_retains_duplicates_and_bad_data(self):
        rows=[{'id':'1','headline':'Headline','true_label':'sports'},
              {'id':'1','headline':'Headline','true_label':'finance'},
              {'id':'3','headline':'','true_label':'other'}]
        issues=p.audit_rows(rows,self.labels)
        self.assertEqual(len(issues),3)
        self.assertIn('duplicate_id',issues[0]['issues'])
        self.assertIn('missing_headline',issues[2]['issues'])
        self.assertEqual(len(rows),3)

    def test_matching_csv_ambiguity(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'a.csv').write_text('id,headline,true_label\n1,hello,sports\n')
            self.assertEqual(p.find_dataset(folder).name,'a.csv')
            (folder/'b.csv').write_text('id,headline,true_label\n2,hello,finance\n')
            with self.assertRaises(ValueError): p.find_dataset(folder)

if __name__=='__main__': unittest.main()
