import 'package:flutter_test/flutter_test.dart';

import 'package:akazi_scroll_app/main.dart';

void main() {
  testWidgets('App renders salary estimator', (WidgetTester tester) async {
    await tester.pumpWidget(const AkaziScrollApp());

    expect(find.text('AKAZI SCROLL — Salary Estimator'), findsOneWidget);
    expect(find.text('Predict'), findsOneWidget);
  });
}
